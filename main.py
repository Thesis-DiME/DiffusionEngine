import json
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import torch
from diffusers import DiffusionPipeline
import hydra


class StableDiffusionImageGenerator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.repo_id = cfg.repo_id
        self.model_name = self.repo_id.replace("/", "-")
        self.device = cfg.device
        self.seed = cfg.seed
        self.cfg_scale = cfg.cfg_scale
        self.num_inference_steps = cfg.num_inference_steps

        self.base_dir = Path(hydra.utils.get_original_cwd())
        self.out_dir = Path(hydra.utils.to_absolute_path(cfg.output_path)) / self.model_name

        self.prompts_path = self.base_dir / "data" / "prompts.txt"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_out_dir = self._prepare_output_directory()

        self.pipe = self._initialize_pipeline()

    def _initialize_pipeline(self):
        torch.manual_seed(self.seed)
        pipe = DiffusionPipeline.from_pretrained(
            self.repo_id,
            torch_dtype=self.torch_dtype,
        )
        return pipe.to(self.device)

    def _prepare_output_directory(self):
        current_out_dir = self.out_dir / self.timestamp
        current_out_dir.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(config=self.cfg, f=str(current_out_dir / "config.yaml"))
        return current_out_dir

    def _load_prompts(self):
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
        

    def generate_images(self):
        prompts = self._load_prompts()
        metadata = []

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        for i, prompt in enumerate(prompts):
            
            output = self.pipe(
                prompt=prompt,
                guidance_scale=self.cfg_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator
            )
            image = output.images[0]

            img_filename = f"image_{i:04d}.png"
            img_path = self.current_out_dir / img_filename
            image.save(img_path)

            metadata.append({
                "image_path": str(img_path.relative_to(self.base_dir)),
                "prompt": prompt,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                print(f"Generated {i+1}/{len(prompts)} images...")

        metadata_path = self.current_out_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

        print(f"\n Export complete: {metadata_path}, {self.current_out_dir}")


@hydra.main(config_path="conf", config_name="sd")
def main(cfg: DictConfig):
    generator = StableDiffusionImageGenerator(cfg)
    generator.generate_images()


if __name__ == "__main__":
    main()
