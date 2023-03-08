import gradio as gr
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from keras_cv import models
from tensorflow import keras

stable_model_list = ["keras-dreambooth/dreambooth_diffusion_model"]

stable_prompt_list = ["a photo of a man.", "a photo of a girl."]

stable_negative_prompt_list = ["bad, ugly", "deformed"]


def keras_stable_diffusion(
    model_path: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: int,
    num_inference_step: int,
    height: int,
    width: int,
):

    with tf.device("/GPU:0"):
        keras.mixed_precision.set_global_policy("mixed_float16")

        sd_dreambooth_model = models.StableDiffusion(
            img_width=height, img_height=width
        )

        db_diffusion_model = from_pretrained_keras(model_path)
        sd_dreambooth_model._diffusion_model = db_diffusion_model

        generated_images = sd_dreambooth_model.text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_inference_step,
            unconditional_guidance_scale=guidance_scale,
        )

    return generated_images


def keras_stable_diffusion_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                keras_text2image_model_path = gr.Dropdown(
                    choices=stable_model_list,
                    value=stable_model_list[0],
                    label="Text-Image Model Id",
                )

                keras_text2image_prompt = gr.Textbox(
                    lines=1, value=stable_prompt_list[0], label="Prompt"
                )

                keras_text2image_negative_prompt = gr.Textbox(
                    lines=1,
                    value=stable_negative_prompt_list[0],
                    label="Negative Prompt",
                )

                with gr.Accordion("Advanced Options", open=False):
                    keras_text2image_guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7.5,
                        label="Guidance Scale",
                    )

                    keras_text2image_num_inference_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Num Inference Step",
                    )

                    keras_text2image_height = gr.Slider(
                        minimum=128,
                        maximum=1280,
                        step=32,
                        value=512,
                        label="Image Height",
                    )

                    keras_text2image_width = gr.Slider(
                        minimum=128,
                        maximum=1280,
                        step=32,
                        value=768,
                        label="Image Height",
                    )

                keras_text2image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Gallery(label="Output")

        keras_text2image_predict.click(
            fn=keras_stable_diffusion,
            inputs=[
                keras_text2image_model_path,
                keras_text2image_prompt,
                keras_text2image_negative_prompt,
                keras_text2image_guidance_scale,
                keras_text2image_num_inference_step,
                keras_text2image_height,
                keras_text2image_width,
            ],
            outputs=output_image,
        )
