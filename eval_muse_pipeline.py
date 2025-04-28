import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
from diffusers import StableDiffusionPipeline
import hydra


class EvalMuseDiffusionPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prompts_path = Path(hydra.utils.to_absolute_path(cfg.prompts_file))
        self.repo_id = cfg.repo_id
        self.model_name = cfg.model_name or self.repo_id.replace("/", "-")
        self.base_dir = Path(hydra.utils.get_original_cwd())
        self.out_dir = self.base_dir / "data" / "generated_images" / self.model_name
        self.device = cfg.device
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = self._initialize_pipeline()

    def _initialize_pipeline(self):
        return StableDiffusionPipeline.from_pretrained(
            self.repo_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

    def _prepare_output_directory(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing_folders = [d for d in self.out_dir.iterdir() if d.is_dir()]
        folder_number = len(existing_folders)
        current_out_dir = self.out_dir / str(folder_number)
        current_out_dir.mkdir(parents=True, exist_ok=True)
        return current_out_dir

    def _load_prompts(self):
        with open(self.prompts_path, "r") as f:
            return json.load(f)

    def generate_images(self):
        eval_muse_metadata = self._load_prompts()
        current_out_dir = self._prepare_output_directory()
        metadata = []

        for i, item in enumerate(eval_muse_metadata[:10]):
            prompt = item["prompt"]
            image = self.pipe(prompt=prompt).images[0]
            img_path = current_out_dir / f"image_{i}.png"
            image.save(img_path)
            item["img_path"] = str(img_path.resolve())
            metadata.append(item)

        metadata_path = current_out_dir / "metadata.json"
        with open(metadata_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)


@hydra.main(config_path="conf", config_name="eval_muse")
def main(cfg: DictConfig):
    pipeline = EvalMuseDiffusionPipeline(cfg)
    pipeline.generate_images()


if __name__ == "__main__":
    main()
