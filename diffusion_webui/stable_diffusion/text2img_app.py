import gradio as gr
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "sd-dreambooth-library/disco-diffusion-style",
    "prompthero/openjourney-v2",
    "andite/anything-v4.0",
    "Lykon/DreamShaper",
    "nitrosocke/Nitro-Diffusion",
    "dreamlike-art/dreamlike-diffusion-1.0",
]

stable_prompt_list = ["a photo of a man.", "a photo of a girl."]

stable_negative_prompt_list = ["bad, ugly", "deformed"]


def stable_diffusion_text2img(
    model_path: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: int,
    num_inference_step: int,
    height: int,
    width: int,
):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, safety_checker=None, torch_dtype=torch.float16
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

    return images[0]


def stable_diffusion_text2img_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                text2image_model_path = gr.Dropdown(
                    choices=stable_model_list,
                    value=stable_model_list[0],
                    label="Text-Image Model Id",
                )

                text2image_prompt = gr.Textbox(
                    lines=1, value=stable_prompt_list[0], label="Prompt"
                )

                text2image_negative_prompt = gr.Textbox(
                    lines=1,
                    value=stable_negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
                    text2image_guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7.5,
                        label="Guidance Scale",
                    )

                    text2image_num_inference_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Num Inference Step",
                    )

                    text2image_height = gr.Slider(
                        minimum=128,
                        maximum=1280,
                        step=32,
                        value=512,
                        label="Image Height",
                    )

                    text2image_width = gr.Slider(
                        minimum=128,
                        maximum=1280,
                        step=32,
                        value=768,
                        label="Image Width",
                    )

                text2image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Image(label="Output")

        gr.Examples(
            examples=[
                [
                    stable_model_list[0],
                    stable_prompt_list[0],
                    stable_negative_prompt_list[0],
                    7.5,
                    50,
                    512,
                    768,
                ]
            ],
            inputs=[
                text2image_model_path,
                text2image_prompt,
                text2image_negative_prompt,
                text2image_guidance_scale,
                text2image_num_inference_step,
                text2image_height,
                text2image_width,
            ],
            outputs=[output_image],
            cache_examples=False,
            fn=stable_diffusion_text2img,
            label="Text2Image Example",
        )
        text2image_predict.click(
            fn=stable_diffusion_text2img,
            inputs=[
                text2image_model_path,
                text2image_prompt,
                text2image_negative_prompt,
                text2image_guidance_scale,
                text2image_num_inference_step,
                text2image_height,
                text2image_width,
            ],
            outputs=output_image,
        )
