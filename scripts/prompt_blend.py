import modules.scripts as scripts
import gradio as gr
import torch

from modules.processing import fix_seed, process_images

class Script(scripts.Script):

    def title(self):
        return "Prompt blend"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        start_prompt = gr.Textbox(label="Start prompt", lines=1)
        end_prompt = gr.Textbox(label="End prompt", lines=1)
            
        with gr.Row():
            start_percent = gr.Number(label="Start percent", value=0.0)
            end_percent = gr.Number(label="End percent", value=1.0)
            prompt_usage = gr.Dropdown(label="Main prompt", choices=["Ignore","Prefix","Suffix"], value="Ignore")

        notes = gr.HTML(value="<div style='margin:15px'>Note: Use a higher batch count for more frames. All images use the same seed (even in batches as long as you use a non-'a' sampler).</div>")

        return [start_prompt, end_prompt, start_percent, end_percent, prompt_usage, notes]

    def run(self, p, start_prompt, end_prompt, start_percent, end_percent, prompt_usage, notes):

        if prompt_usage == "Prefix":
            start_prompt = f"{p.prompt} {start_prompt}"
            end_prompt = f"{p.prompt} {end_prompt}"

        if prompt_usage == "Suffix":
            start_prompt = f"{start_prompt} {p.prompt}"
            end_prompt = f"{end_prompt} {p.prompt}"

        self.total = p.n_iter*p.batch_size 
        self.iteration = 0

        self.original_sample = p.sample

        def sample_extra (x, conditioning, unconditional_conditioning):

            start_cond = p.sd_model.get_learned_conditioning([start_prompt])
            end_cond = p.sd_model.get_learned_conditioning([end_prompt])

            conds = []

            for _ in range(p.batch_size):
                blend_percent = self.iteration/(self.total-1) if self.total > 1 else 0.5 #if we are only generating one image, create a 50% blend between start and end prompt 
                blend_percent = start_percent + blend_percent * (end_percent-start_percent) #remap percent to within a specific range
                conds.append(torch.lerp(start_cond,end_cond,blend_percent)) #blend/lerp between the actual conditioning tensors 

                self.iteration += 1 #we want to blend smoothly within the batches 

            conditioning = torch.cat(conds)

            samples_ddim = self.original_sample(x, conditioning, unconditional_conditioning)
            
            return samples_ddim

        p.sample = sample_extra
    
        p.prompt = f"{start_prompt} to {end_prompt}"

        fix_seed(p) #use the specified seed or get a random one if needed
        p.seed = p.n_iter*p.batch_size * [int(p.seed)] #force the seed to stay the same for each iteration

        p.extra_generation_params = {"Start percent":start_percent,"End percent":end_percent}

        processed = process_images(p)

        return processed

