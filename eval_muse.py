import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
from diffusers import StableDiffusionPipeline
import hydra


@hydra.main(config_path="conf", config_name="eval_muse")
def main(cfg: DictConfig):
    prompts_path = Path(hydra.utils.to_absolute_path(cfg.prompts_file))
    with open(prompts_path, "r") as f:
        eval_muse_metadata = json.load(f)

    repo_id = cfg.repo_id
    model_name = cfg.model_name or repo_id.replace('/', '-')

    base_dir = Path(hydra.utils.get_original_cwd())
    out_dir = base_dir / "data" / "generated_images" / model_name
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    existing_folders = [d for d in out_dir.iterdir() if d.is_dir()]
    folder_number = len(existing_folders)
    
    current_out_dir = out_dir / str(folder_number)
    current_out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    metadata = []
    for i, item in enumerate(eval_muse_metadata[:10]):
        prompt = item['prompt']
        image = pipe(prompt=prompt).images[0]
        img_path = current_out_dir / f"image_{i}.png"
        image.save(img_path)
        item['img_path'] = str(img_path.resolve())
        metadata.append(item)

    metadata_path = current_out_dir / "metadata.json"
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


if __name__ == "__main__":
    main()