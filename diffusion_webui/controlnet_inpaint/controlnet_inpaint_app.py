import gradio as gr
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from PIL import Image

from diffusion_webui.controlnet.controlnet_canny import controlnet_canny
from diffusion_webui.controlnet_inpaint.pipeline_stable_diffusion_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)

stable_inpaint_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
]

controlnet_model_list = [
    "lllyasviel/sd-controlnet-canny",
]

prompt_list = [
    "a red panda sitting on a bench",
]

negative_prompt_list = [
    "bad, ugly",
]


def load_img(image_path: str):
    image = Image.open(image_path)
    image = np.array(image)
    image = Image.fromarray(image)

    return image


def stable_diffusion_inpiant_controlnet_canny(
    normal_image_path: str,
    stable_model_path: str,
    controlnet_model_path: str,
    prompt: str,
    negative_prompt: str,
    controlnet_conditioning_scale: str,
    guidance_scale: int,
    num_inference_steps: int,
):
    pil_image = Image.open(normal_image_path)
    normal_image = pil_image["image"].convert("RGB").resize((512, 512))
    mask_image = pil_image["mask"].convert("RGB").resize((512, 512))

    # normal_image = load_img(normal_image_path)
    # mask_image = load_img(mask_image_path)

    controlnet, control_image = controlnet_canny(
        image_path=normal_image_path,
        controlnet_model_path=controlnet_model_path,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path=stable_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.manual_seed(0)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        image=normal_image,
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        mask_image=mask_image,
    ).images

    return output[0]


def stable_diffusion_inpiant_controlnet_canny_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                inpaint_image_file = gr.Image(
                    source="upload",
                    tool="sketch",
                    elem_id="image_upload",
                    type="filepath",
                    label="Upload",
                )

                inpaint_model_id = gr.Dropdown(
                    choices=stable_inpaint_model_list,
                    value=stable_inpaint_model_list[0],
                    label="Inpaint Model Id",
                )

                inpaint_controlnet_model_id = gr.Dropdown(
                    choices=controlnet_model_list,
                    value=controlnet_model_list[0],
                    label="ControlNet Model Id",
                )

                inpaint_prompt = gr.Textbox(
                    lines=1, value=prompt_list[0], label="Prompt"
                )

                inpaint_negative_prompt = gr.Textbox(
                    lines=1,
                    value=negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
                    controlnet_conditioning_scale = gr.Slider(
                        minimum=0.1,
                        maximum=1,
                        step=0.1,
                        value=0.5,
                        label="ControlNet Conditioning Scale",
                    )

                    inpaint_guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7.5,
                        label="Guidance Scale",
                    )

                    inpaint_num_inference_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Num Inference Step",
                    )

                inpaint_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image(label="Outputs")

        inpaint_predict.click(
            fn=stable_diffusion_inpiant_controlnet_canny,
            inputs=[
                inpaint_image_file,
                inpaint_model_id,
                inpaint_controlnet_model_id,
                inpaint_prompt,
                inpaint_negative_prompt,
                controlnet_conditioning_scale,
                inpaint_guidance_scale,
                inpaint_num_inference_step,
            ],
            outputs=output_image,
        )
