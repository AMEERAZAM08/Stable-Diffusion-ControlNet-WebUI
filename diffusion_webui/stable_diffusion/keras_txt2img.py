import gradio as gr
from huggingface_hub import from_pretrained_keras
from keras_cv import models
from tensorflow import keras

keras_model_list = [
    "keras-dreambooth/keras_diffusion_lowpoly_world",
    "keras-dreambooth/keras-diffusion-traditional-furniture",
]

stable_prompt_list = [
    "photo of lowpoly_world",
    "photo of traditional_furniture",
]

stable_negative_prompt_list = ["bad, ugly", "deformed"]

keras.mixed_precision.set_global_policy("mixed_float16")
dreambooth_model = models.StableDiffusion(
    img_width=512,
    img_height=512,
    jit_compile=True,
)


def keras_stable_diffusion(
    model_path: str,
    prompt: str,
    negative_prompt: str,
    num_imgs_to_gen: int,
    num_steps: int,
):
    """
    This function is used to generate images using our fine-tuned keras dreambooth stable diffusion model.
    Args:
        prompt (str): The text input given by the user based on which images will be generated.
        num_imgs_to_gen (int): The number of images to be generated using given prompt.
        num_steps (int): The number of denoising steps
    Returns:
        generated_img (List): List of images that were generated using the model
    """
    loaded_diffusion_model = from_pretrained_keras(model_path)
    dreambooth_model._diffusion_model = loaded_diffusion_model

    generated_img = dreambooth_model.text_to_image(
        prompt,
        negative_prompt=negative_prompt,
        batch_size=num_imgs_to_gen,
        num_steps=num_steps,
    )

    return generated_img


def keras_stable_diffusion_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                keras_text2image_model_path = gr.Dropdown(
                    choices=keras_model_list,
                    value=keras_model_list[0],
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

                keras_text2image_predict = gr.Button(value="Generator")

            with gr.Column():
                output_image = gr.Gallery(label="Outputs").style(grid=(1, 2))

        gr.Examples(
            fn=keras_stable_diffusion,
            inputs=[
                keras_text2image_model_path,
                keras_text2image_prompt,
                keras_text2image_negative_prompt,
                keras_text2image_guidance_scale,
                keras_text2image_num_inference_step,
            ],
            outputs=[output_image],
            examples=[
                [
                    keras_model_list[0],
                    stable_prompt_list[0],
                    stable_negative_prompt_list[0],
                    7.5,
                    50,
                    512,
                    512,
                ],
            ],
            label="Keras Stable Diffusion Example",
            cache_examples=False,
        )

        keras_text2image_predict.click(
            fn=keras_stable_diffusion,
            inputs=[
                keras_text2image_model_path,
                keras_text2image_prompt,
                keras_text2image_negative_prompt,
                keras_text2image_guidance_scale,
                keras_text2image_num_inference_step,
            ],
            outputs=output_image,
        )
