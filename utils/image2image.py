from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from IPython.display import display
from PIL import Image
import torch

def stable_diffusion_img2img(
    model_path:str,
    image_path:str,
    prompt:str,
    negative_prompt:str,
    num_samples:int,
    guidance_scale:int,
    num_inference_step:int,
    ):

    image = Image.open(image_path)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path, 
        safety_checker=None, 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    output = pipe(
        prompt = prompt,
        image = image,
        negative_prompt = negative_prompt,
        num_images_per_prompt = num_samples,
        num_inference_steps = num_inference_step,
        guidance_scale = guidance_scale,
    ).images

    return output
