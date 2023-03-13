import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
]

controlnet_canny_model_list = [
    "lllyasviel/sd-controlnet-canny",
    "thibaud/controlnet-sd21-canny-diffusers",
]


stable_prompt_list = ["a photo of a man.", "a photo of a girl."]

stable_negative_prompt_list = ["bad, ugly", "deformed"]

data_list = [
    "data/test.png",
]


def controlnet_canny(
    image_path: str,
    controlnet_model_path: str,
):
    image = Image.open(image_path)
    image = np.array(image)

    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path, torch_dtype=torch.float16
    )
    return controlnet, image


def stable_diffusion_controlnet_canny(
    image_path: str,
    stable_model_path: str,
    controlnet_model_path: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: int,
    num_inference_step: int,
):

    controlnet, image = controlnet_canny(
        image_path=image_path, controlnet_model_path=controlnet_model_path
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=stable_model_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    output = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
    ).images

    return output[0]


def stable_diffusion_controlnet_canny_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                controlnet_canny_image_file = gr.Image(
                    type="filepath", label="Image"
                )

                controlnet_canny_stable_model_id = gr.Dropdown(
                    choices=stable_model_list,
                    value=stable_model_list[0],
                    label="Stable Model Id",
                )

                controlnet_canny_model_id = gr.Dropdown(
                    choices=controlnet_canny_model_list,
                    value=controlnet_canny_model_list[0],
                    label="Controlnet Model Id",
                )

                controlnet_canny_prompt = gr.Textbox(
                    lines=1, value=stable_prompt_list[0], label="Prompt"
                )

                controlnet_canny_negative_prompt = gr.Textbox(
                    lines=1,
                    value=stable_negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
                    controlnet_canny_guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7.5,
                        label="Guidance Scale",
                    )

                    controlnet_canny_num_inference_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Num Inference Step",
                    )

                controlnet_canny_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image(label="Output")

        gr.Examples(
            fn=stable_diffusion_controlnet_canny,
            examples=[
                [
                    data_list[0],
                    stable_model_list[0],
                    controlnet_canny_model_list[0],
                    stable_prompt_list[0],
                    stable_negative_prompt_list[0],
                    7.5,
                    50,
                ]
            ],
            inputs=[
                controlnet_canny_image_file,
                controlnet_canny_stable_model_id,
                controlnet_canny_model_id,
                controlnet_canny_prompt,
                controlnet_canny_negative_prompt,
                controlnet_canny_guidance_scale,
                controlnet_canny_num_inference_step,
            ],
            outputs=[output_image],
            cache_examples=False,
            label="Controlnet Canny Example",
        )

        controlnet_canny_predict.click(
            fn=stable_diffusion_controlnet_canny,
            inputs=[
                controlnet_canny_image_file,
                controlnet_canny_stable_model_id,
                controlnet_canny_model_id,
                controlnet_canny_prompt,
                controlnet_canny_negative_prompt,
                controlnet_canny_guidance_scale,
                controlnet_canny_num_inference_step,
            ],
            outputs=[output_image],
        )
