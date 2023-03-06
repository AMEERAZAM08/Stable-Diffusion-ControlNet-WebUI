from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

def stable_diffusion_text2img(
    model_path:str,
    prompt:str,
    negative_prompt:str,
    guidance_scale:int,
    num_inference_step:int,
    height:int,
    width:int,
    ):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        safety_checker=None, 
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
    ).images

    return images
