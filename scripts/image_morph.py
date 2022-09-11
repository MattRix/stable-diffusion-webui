import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Image Morph"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        start_prompt = gr.Textbox(label="Start Prompt", visible=True, lines=1)
        end_prompt = gr.Textbox(label="End Prompt", visible=True, lines=1)
        return [start_prompt,end_prompt]

    def run(self, p, start_prompt, end_prompt):

        def override(iter):
            start_cond = p.sd_model.get_learned_conditioning([start_prompt]*p.batch_size)
            end_cond = p.sd_model.get_learned_conditioning([end_prompt]*p.batch_size)

            blend_percent = iter/(p.n_iter-1) if p.n_iter > 1 else 0
            mix_cond = start_cond * (1.0-blend_percent) + end_cond * blend_percent

            return mix_cond

        p.prompt = f"{start_prompt} to {end_prompt}"

        p.seed = p.n_iter*p.batch_size * [123]

        #p.extra_generation_params = {"Start prompt":start_prompt,"End prompt":end_prompt}

        p.cond_override = override
        processed = process_images(p)
        p.cond_override = None

        return processed

