from diffusers import ( StableDiffusionControlNetPipeline, 
                       ControlNetModel, UniPCMultistepScheduler,
                       DDIMScheduler)

from transformers import pipeline
from PIL import Image
import numpy as np
import torch


def controlnet_depth(image_path:str):
    depth_estimator = pipeline('depth-estimation')

    image = Image.open(image_path)
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16
    )

    return controlnet, image

def stable_diffusion_controlnet_depth(
    stable_model_path:str,
    image_path:str,
    prompt:str,
    negative_prompt:str,
    num_samples:int,
    guidance_scale:int,
    num_inference_step:int,
    ):

    controlnet, image = controlnet_depth(image_path=image_path)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=stable_model_path, 
        controlnet=controlnet, 
        safety_checker=None, 
        torch_dtype=torch.float16
    )

    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
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
