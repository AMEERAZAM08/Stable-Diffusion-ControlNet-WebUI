import gradio as gr
import torch
from diffusers import DDIMScheduler, DiffusionPipeline

stable_inpiant_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
]

stable_prompt_list = ["a photo of a man.", "a photo of a girl."]
stable_negative_prompt_list = ["bad, ugly", "deformed"]


def stable_diffusion_inpaint(
    dict: str,
    model_path: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: int,
    num_inference_step: int,
):

    image = dict["image"].convert("RGB").resize((512, 512))
    mask_image = dict["mask"].convert("RGB").resize((512, 512))
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    output = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_step,
        guidance_scale=guidance_scale,
    ).images

    return output[0]


def stable_diffusion_inpaint_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                inpaint_image_file = gr.Image(
                    source="upload",
                    tool="sketch",
                    elem_id="image_upload",
                    type="pil",
                    label="Upload",
                )

                inpaint_model_id = gr.Dropdown(
                    choices=stable_inpiant_model_list,
                    value=stable_inpiant_model_list[0],
                    label="Inpaint Model Id",
                )

                inpaint_prompt = gr.Textbox(
                    lines=1, value=stable_prompt_list[0], label="Prompt"
                )

                inpaint_negative_prompt = gr.Textbox(
                    lines=1,
                    value=stable_negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
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
            fn=stable_diffusion_inpaint,
            inputs=[
                inpaint_image_file,
                inpaint_model_id,
                inpaint_prompt,
                inpaint_negative_prompt,
                inpaint_guidance_scale,
                inpaint_num_inference_step,
            ],
            outputs=output_image,
        )
