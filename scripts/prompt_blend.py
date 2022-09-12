import modules.scripts as scripts
import gradio as gr
import torch

from modules.processing import Processed, fix_seed, process_images
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Prompt blend"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
           
        start_prompt = gr.Textbox(label="Start prompt", lines=1)
        end_prompt = gr.Textbox(label="End prompt", lines=1)
            
        with gr.Row():
            start_percent = gr.Number(label="Start percent", value=0.0)
            end_percent = gr.Number(label="End percent", value=1.0)
            prompt_usage = gr.Dropdown(label="Main prompt", choices=["Ignore","Prefix","Suffix"], value="Ignore")

        return [start_prompt, end_prompt, start_percent, end_percent, prompt_usage]

    def run(self, p, start_prompt, end_prompt, start_percent, end_percent, prompt_usage):

        if prompt_usage == "Prefix":
            start_prompt = f"{p.prompt} {start_prompt}"
            end_prompt = f"{p.prompt} {end_prompt}"

        if prompt_usage == "Suffix":
            start_prompt = f"{start_prompt} {p.prompt}"
            end_prompt = f"{end_prompt} {p.prompt}"
            
        #replace the conditioning with a blend between the start prompt and end prompt conditioning 
        def cond_override(iter,uc,c):
            start_cond = p.sd_model.get_learned_conditioning([start_prompt]*p.batch_size)
            end_cond = p.sd_model.get_learned_conditioning([end_prompt]*p.batch_size)

            #if we are only generating one image, create a 50% blend between start and end prompt
            blend_percent = iter/(p.n_iter-1) if p.n_iter > 1 else 0.5

            #remap percent to within a specific range
            blend_percent = start_percent + blend_percent * (end_percent-start_percent)

            blend_cond = torch.lerp(start_cond,end_cond,blend_percent)

            return (uc,blend_cond)

        p.prompt = f"{start_prompt} to {end_prompt}"

        #keep the seed to stay the same for each iteration
        fix_seed(p)
        p.seed = p.n_iter*p.batch_size * [int(p.seed)]

        p.extra_generation_params = {"Start percent":start_percent,"End percent":end_percent}

        p.cond_override = cond_override
        processed = process_images(p)
        p.cond_override = None

        return processed

