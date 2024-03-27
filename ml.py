from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token_path = Path('Token.txt')
token = token_path.read_text().strip()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# get y
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token=token,
    low_cpu_mem_usage=False
)

pipe.to("cuda")
prompt = "A photograph of fishing"
image = pipe(prompt)["sample"][0]


def obtain_image(
        prompt: str,
        *,
        seed: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5) -> Image:
    generator = None if seed is None else torch.Generator('cuda').manual_seed(seed)

    image: Image = pipe(prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator
                        ).images[0]

    return image
