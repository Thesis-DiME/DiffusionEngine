import os
import json
import torch
from diffusers import StableDiffusionPipeline

def main():
    with open("data/prompts.txt", "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    model_name = "sd1.5"
    repo_id = "runwayml/stable-diffusion-v1-5"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output", model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    existing_folders = [d for d in os.listdir(out_dir)]
    folder_number = len(existing_folders)
    
    out_dir = os.path.join(out_dir, str(folder_number))
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        safety_checker=None,  # Disable safety checker
        requires_safety_checker=False,
    ).to(device)

    metadata = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt=prompt).images[0]
        img_path = os.path.join(out_dir, f"image_{i}.png")
        image.save(img_path)
        
        metadata.append({
            "image_path": os.path.relpath(img_path, base_dir),
            "prompt": prompt
        })

    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()