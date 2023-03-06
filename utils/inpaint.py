from diffusers import DiffusionPipeline, DDIMScheduler
from PIL import Image
import imageio
import torch

# https://huggingface.co/spaces/Manjushri/SD-2.0-Inpainting-CPU/blob/main/app.py

def resize(height,img):
  baseheight = height
  img = Image.open(img)
  hpercent = (baseheight/float(img.size[1]))
  wsize = int((float(img.size[0])*float(hpercent)))
  img = img.resize((wsize,baseheight), Image.Resampling.LANCZOS)
  return img

def img_preprocces(source_img, prompt, negative_prompt):
    imageio.imwrite("data.png", source_img["image"])
    imageio.imwrite("data_mask.png", source_img["mask"])
    src = resize(512, "data.png")
    src.save("src.png")
    mask = resize(512, "data_mask.png")  
    mask.save("mask.png")
    return src, mask

def stable_diffusion_inpaint(
    image_path:str,
    model_path:str,
    prompt:str,
    negative_prompt:str,
    guidance_scale:int,
    num_inference_step:int,
    ):

    image, mask_image = img_preprocces(image_path, prompt, negative_prompt)
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    output = pipe(
        prompt = prompt,
        image = image,
        mask_image=mask_image,
        negative_prompt = negative_prompt,
        num_inference_steps = num_inference_step,
        guidance_scale = guidance_scale,
    ).images

    return output
