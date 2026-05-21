from __future__ import annotations

import argparse
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import textwrap
import time
from typing import Any

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedType
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from eval.hf_inference_3d import (
    NumpyEncoder,
    open6dor_compute_score,
    parse_points_from_output,
    process_vision_info,
)


def load_image(row: dict[str, Any], image_root: str) -> Image.Image:
    source = row.get("rgb_image_path")
    if not source:
        images = row.get("images") or []
        if images:
            source = images[0]

    if isinstance(source, str):
        normalized = source.replace("\\", "/")
        candidates = [Path(normalized)]
        if image_root:
            candidates.append(Path(image_root) / normalized)
        for candidate in candidates:
            if candidate.exists():
                return Image.open(candidate).convert("RGB")

    image_data = row.get("image")
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, (bytes, bytearray)):
        return Image.open(BytesIO(image_data)).convert("RGB")

    raise FileNotFoundError(f"Cannot locate image for sample {row.get('sample_id')}: {source}")


def instruction_for(row: dict[str, Any], condition: str) -> str:
    if condition == "baseline":
        return row["baseline_instruction"]
    if condition == "ids_oracle":
        return row["ids_oracle_instruction"]
    if condition == "oracle_only":
        return row["ids_oracle_instruction"] if row.get("oracle_available") else row["baseline_instruction"]
    raise ValueError(f"Unsupported condition: {condition}")


def process_sample(
    row: dict[str, Any],
    model,
    processor,
    device,
    reasoning_model: bool,
    instruct_following: str,
    gen_args: dict[str, Any],
    condition: str,
    image_root: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    image = load_image(row, image_root)
    instruction = instruction_for(row, condition)
    question = instruction + "\n" + instruct_following
    question = textwrap.dedent(question).strip()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_args)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    original_predicted = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    predicted = original_predicted
    if reasoning_model:
        import re

        match = re.search(r"<answer>(.*?)</answer>", predicted)
        predicted = match.group(1).strip() if match else ""

    points = parse_points_from_output(predicted)
    score = open6dor_compute_score(predicted, row["answer"], logger)

    return {
        "sample_id": row.get("sample_id"),
        "sample_index": row.get("sample_index"),
        "condition": condition,
        "question": question,
        "baseline_instruction": row.get("baseline_instruction"),
        "ids_oracle_instruction": row.get("ids_oracle_instruction"),
        "ids_status": row.get("ids_status"),
        "oracle_available": row.get("oracle_available"),
        "oracle_quality": row.get("oracle_quality"),
        "oracle_clarification": row.get("oracle_clarification"),
        "predicted": predicted,
        "original_predicted": original_predicted,
        "points": points,
        "ground_truth": row.get("answer"),
        "accuracy_score": float(score) if isinstance(score, (np.number, np.ndarray)) else score,
        "position_tag": row.get("position_tag", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-json", default="output_results/ids_oracle_clarification_3d.json")
    parser.add_argument("--condition", choices=["baseline", "ids_oracle", "oracle_only"], default="ids_oracle")
    parser.add_argument("--model-path", default="IffYuan/Embodied-R1-3B-v1")
    parser.add_argument("--model-name", default="Embodied-R1-3B")
    parser.add_argument("--task-name", default="Open6dor-IDS-Oracle")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="logs/results")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ambiguous-only", action="store_true")
    parser.add_argument("--use-flash-attention", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"logs/inference_{args.task_name}_{args.condition}_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - Process %(process)d - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_name)],
    )
    logger = logging.getLogger(f"{args.task_name}_{args.condition}")

    with open(args.oracle_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    rows = report["samples"]
    if args.ambiguous_only:
        rows = [row for row in rows if row.get("ids_status") == "ambiguous"]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(42)

    if accelerator.num_processes > 1:
        local_device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
    else:
        local_device = device
        device_map = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2" if args.use_flash_attention else None,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=3110400,
        min_pixels=256 * 28 * 28,
    )

    if accelerator.num_processes > 1:
        if accelerator.distributed_type == DistributedType.FSDP:
            model = accelerator.prepare(model)
        else:
            model = accelerator.prepare_model(model, evaluation_mode=True)

    gen_args = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.05,
        "do_sample": False,
    }
    instruct_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
        r"The answer consists only of several coordinate points, with the overall format being: "
        r"<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
    )

    process_idx = accelerator.process_index
    num_processes = accelerator.num_processes
    samples_per_process = len(rows) // num_processes
    start_idx = process_idx * samples_per_process
    end_idx = start_idx + samples_per_process if process_idx < num_processes - 1 else len(rows)

    local_results = []
    iterator = tqdm(range(start_idx, end_idx), desc=f"Process {process_idx}") if accelerator.is_main_process else range(start_idx, end_idx)
    for idx in iterator:
        result = process_sample(
            rows[idx],
            model,
            processor,
            local_device,
            True,
            instruct_following,
            gen_args,
            args.condition,
            args.image_root,
            logger,
        )
        local_results.append(result)

    all_results = accelerator.gather_for_metrics(local_results)
    if accelerator.is_main_process:
        final_results = [item for sublist in all_results for item in sublist] if isinstance(all_results[0], list) else all_results
        total = len(final_results)
        correct = sum(1 for row in final_results if row.get("accuracy_score", 0) > 0)
        accuracy = correct / total if total else 0
        summary = {
            "condition": args.condition,
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "oracle_json": args.oracle_json,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"{args.task_name}_{args.model_name}_{args.condition}_{current_time}.json",
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": final_results}, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
        logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
