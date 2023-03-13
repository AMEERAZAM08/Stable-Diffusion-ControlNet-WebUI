import gradio as gr

from diffusion_webui.helpers import (
    keras_stable_diffusion_app,
    stable_diffusion_controlnet_canny_app,
    stable_diffusion_controlnet_depth_app,
    stable_diffusion_controlnet_hed_app,
    stable_diffusion_controlnet_mlsd_app,
    stable_diffusion_controlnet_pose_app,
    stable_diffusion_controlnet_scribble_app,
    stable_diffusion_controlnet_seg_app,
    stable_diffusion_img2img_app,
    stable_diffusion_inpaint_app,
    stable_diffusion_text2img_app,
)

app = gr.Blocks()
with app:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Stable Diffusion + ControlNet + Keras Diffusion WebUI
        </h1>
        """
    )
    gr.Markdown(
        """
        <h4 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h4>
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("Text2Img"):
                stable_diffusion_text2img_app()
            with gr.Tab("Img2Img"):
                stable_diffusion_img2img_app()
            with gr.Tab("Inpaint"):
                stable_diffusion_inpaint_app()

            with gr.Tab("ControlNet"):
                with gr.Tab("Canny"):
                    stable_diffusion_controlnet_canny_app()
                with gr.Tab("Depth"):
                    stable_diffusion_controlnet_depth_app()
                with gr.Tab("HED"):
                    stable_diffusion_controlnet_hed_app()
                with gr.Tab("MLSD"):
                    stable_diffusion_controlnet_mlsd_app()
                with gr.Tab("Pose"):
                    stable_diffusion_controlnet_pose_app()
                with gr.Tab("Seg"):
                    stable_diffusion_controlnet_seg_app()
                with gr.Tab("Scribble"):
                    stable_diffusion_controlnet_scribble_app()

            with gr.Tab("Keras Diffusion"):
                keras_diffusion_app = keras_stable_diffusion_app()

app.launch(debug=True)
