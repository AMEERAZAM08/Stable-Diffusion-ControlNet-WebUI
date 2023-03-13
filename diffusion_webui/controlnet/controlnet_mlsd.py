import gradio as gr
import torch
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
]

stable_prompt_list = ["a photo of a man.", "a photo of a girl."]

stable_negative_prompt_list = ["bad, ugly", "deformed"]

data_list = [
    "data/test.png",
]


def controlnet_mlsd(image_path: str):
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    image = Image.open(image_path)
    image = mlsd(image)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd",
        torch_dtype=torch.float16,
    )

    return controlnet, image


def stable_diffusion_controlnet_mlsd(
    image_path: str,
    model_path: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: int,
    num_inference_step: int,
):

    controlnet, image = controlnet_mlsd(image_path=image_path)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
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


def stable_diffusion_controlnet_mlsd_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                controlnet_mlsd_image_file = gr.Image(
                    type="filepath", label="Image"
                )

                controlnet_mlsd_model_id = gr.Dropdown(
                    choices=stable_model_list,
                    value=stable_model_list[0],
                    label="Stable Model Id",
                )

                controlnet_mlsd_prompt = gr.Textbox(
                    lines=1, value=stable_prompt_list[0], label="Prompt"
                )

                controlnet_mlsd_negative_prompt = gr.Textbox(
                    lines=1,
                    value=stable_negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
                    controlnet_mlsd_guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7.5,
                        label="Guidance Scale",
                    )

                    controlnet_mlsd_num_inference_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Num Inference Step",
                    )

                controlnet_mlsd_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image(label="Output")

        gr.Examples(
            fn=stable_diffusion_controlnet_mlsd,
            examples=[
                [
                    data_list[0],
                    stable_model_list[0],
                    stable_prompt_list[0],
                    stable_negative_prompt_list[0],
                    7.5,
                    50,
                ]
            ],
            inputs=[
                controlnet_mlsd_image_file,
                controlnet_mlsd_model_id,
                controlnet_mlsd_prompt,
                controlnet_mlsd_negative_prompt,
                controlnet_mlsd_guidance_scale,
                controlnet_mlsd_num_inference_step,
            ],
            outputs=[output_image],
            label="ControlNet-MLSD Example",
            cache_examples=False,
        )

        controlnet_mlsd_predict.click(
            fn=stable_diffusion_controlnet_mlsd,
            inputs=[
                controlnet_mlsd_image_file,
                controlnet_mlsd_model_id,
                controlnet_mlsd_prompt,
                controlnet_mlsd_negative_prompt,
                controlnet_mlsd_guidance_scale,
                controlnet_mlsd_num_inference_step,
            ],
            outputs=output_image,
        )
