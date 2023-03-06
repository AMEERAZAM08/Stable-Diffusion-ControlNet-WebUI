from diffusers import ( StableDiffusionControlNetPipeline, 
                       ControlNetModel, UniPCMultistepScheduler)

from controlnet_aux import OpenposeDetector

from PIL import Image
import torch


def controlnet_pose(image_path:str):
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    image = Image.open(image_path)
    image = openpose(image)

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", 
        torch_dtype=torch.float16
    )

    return controlnet, image

def stable_diffusion_controlnet_pose(
    stable_model_path:str,
    image_path:str,
    prompt:str,
    negative_prompt:str,
    num_samples:int,
    guidance_scale:int,
    num_inference_step:int,
    ):

    controlnet, image = controlnet_pose(image_path=image_path)

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
