import json
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
import os


@hydra.main(config_path="conf", config_name="sd")
def main(cfg: DictConfig):
    base_dir = get_original_cwd()

    # Load prompts
    prompts_path = os.path.join(base_dir, cfg.prompts_path)
    with open(prompts_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    model_name = "sd1.5"
    out_dir = os.path.join(base_dir, cfg.output_path, model_name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(cfg.device)

    metadata = []
    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            guidance_scale=cfg.cfg_scale,
            num_inference_steps=cfg.num_inference_steps,
        ).images[0]
        file_name = f"image_{i}.png"
        img_path = os.path.join(out_dir, file_name)
        image.save(img_path)

        metadata.append({"img_path": file_name, "prompt": prompt})

    # Save metadata to JSON
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
