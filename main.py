import os
import json
import torch
import argparse
from datetime import datetime

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    with open("./data/prompts.txt", "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    model_name = "sd1.5"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.output_path, model_name)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    metadata = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt=prompt).images[0]
        img_path = os.path.join(out_dir, f"image_{i}.png")
        image.save(img_path)

        metadata.append(
            {"image_path": os.path.relpath(img_path, base_dir), "prompt": prompt}
        )

    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
