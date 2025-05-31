import json
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
import os


@hydra.main(config_path="conf", config_name="sd")
def main(cfg: DictConfig):
    base_dir = get_original_cwd()

    prompts_path = os.path.join(base_dir, cfg.prompts_path)
    _, ext = os.path.splitext(prompts_path)

    # Load prompts
    if ext == ".json":
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
            assert isinstance(prompts, list), "JSON must be a list of dicts"
            assert all("prompt" in item for item in prompts), (
                "Each item must contain a 'prompt' key"
            )
    elif ext == ".txt":
        with open(prompts_path, "r") as f:
            prompts = [{"prompt": line.strip()} for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    os.makedirs(os.path.join(base_dir, cfg.output_path), exist_ok=True)
    out_dir = os.path.join(
        base_dir, cfg.output_path, cfg.model_id.replace("/", "_"), cfg.run_id
    )
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = torch.float16 if cfg.device == "cuda" else torch.float32

    # if cfg.model_id == "sd-legacy/stable-diffusion-v1-5":
    #     vae = AutoencoderKL.from_pretrained(
    #         "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    #     )
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         cfg.model_id,
    #         vae=vae,
    #         torch_dtype=torch.float16,  # torch_dtype,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     ).to(cfg.device)
    #     print('vae')
    # else:
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,  # torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(cfg.device)

    metadata = []
    for i, item in enumerate(prompts):
        prompt_text = item["prompt"]

        image = pipe(
            prompt=prompt_text,
            guidance_scale=cfg.cfg_scale,
            num_inference_steps=cfg.num_inference_steps,
            num_images_per_prompt=1,
        ).images[0]

        file_name = f"image_{i}.png"
        img_path = os.path.join(out_dir, file_name)
        image.save(img_path)

        result_entry = dict(
            item
        )  # preserve all JSON keys (e.g. custom tags, attributes)
        result_entry["img_path"] = file_name
        metadata.append(result_entry)

    # Save metadata to JSON
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Saved {len(metadata)} images and metadata to {out_dir}")


if __name__ == "__main__":
    main()
