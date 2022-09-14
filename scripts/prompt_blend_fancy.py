from pickletools import optimize
from re import I
import imageio
import modules.scripts as scripts
import gradio as gr
import torch

from modules.processing import fix_seed, process_images, create_random_tensors, opt_C, opt_f

class Script(scripts.Script):

    def title(self):
        return "Prompt blend fancy"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):

        start_prompt = gr.Textbox(label="Start prompt", lines=1)
        end_prompt = gr.Textbox(label="End prompt", lines=1)
            
        with gr.Row():
            start_percent = gr.Number(label="Start percent", value=0.0)
            end_percent = gr.Number(label="End percent", value=1.0)
            prompt_usage = gr.Dropdown(label="Main prompt", choices=["Ignore","Prefix","Suffix"], value="Ignore")
            frame_count = gr.Number(label="Batch count override", value=-1)

        with gr.Row():
            start_seed = gr.Number(label="Start seed", value=-1)
            end_seed = gr.Number(label="End seed", value=-1)
            make_gif = gr.Checkbox(label="Make gif", value=True)

        notes = gr.HTML(value="<div style='margin:15px'>Note: Use a higher batch count for more frames. All images use the same seed (even in batches as long as you use a non-'a' sampler).</div>")

        return [start_prompt, end_prompt, start_percent, end_percent, prompt_usage, frame_count, start_seed, end_seed, make_gif, notes]

    def run(self, p, start_prompt, end_prompt, start_percent, end_percent, prompt_usage, frame_count, start_seed, end_seed, make_gif, notes):

        if prompt_usage == "Prefix":
            start_prompt = f"{p.prompt} {start_prompt}"
            end_prompt = f"{p.prompt} {end_prompt}"

        if prompt_usage == "Suffix":
            start_prompt = f"{start_prompt} {p.prompt}"
            end_prompt = f"{end_prompt} {p.prompt}"

        if int(frame_count) > 0:
            p.n_iter = int(frame_count) # we want to be able to make more than 16 frames for animations

        self.original_sample = p.sample

        self.total = p.n_iter*p.batch_size 
        self.iteration = 0
        
        self.should_blend_noise = (start_seed != end_seed) #don't lerp if the seeds are the same, the slerp won't won't work

        start_noise = create_random_tensors([opt_C, p.width // opt_f, p.height // opt_f], [start_seed])
        end_noise = create_random_tensors([opt_C, p.width // opt_f, p.height // opt_f], [end_seed])

        def sample_extra (x, conditioning, unconditional_conditioning):

            start_cond = p.sd_model.get_learned_conditioning([start_prompt])
            end_cond = p.sd_model.get_learned_conditioning([end_prompt])

            noises = []
            conds = []

            for _ in range(p.batch_size):
                blend_percent = self.iteration/(self.total-1) if self.total > 1 else 0.5 # if we are only generating one image, create a 50% blend between start and end prompt 
                blend_percent = start_percent + blend_percent * (end_percent-start_percent) # remap percent to within a specific range
                conds.append(torch.lerp(start_cond,end_cond,blend_percent)) # blend/lerp between the actual conditioning tensors 

                if self.should_blend_noise:
                    noises.append(slerp(start_noise,end_noise,blend_percent))

                self.iteration += 1 # we want to blend smoothly within each batch

            conditioning = torch.cat(conds)
            noise = torch.cat(noises)

            samples_ddim = self.original_sample(noise, conditioning, unconditional_conditioning)

            return samples_ddim

        p.sample = sample_extra
    
        p.prompt = f"{start_prompt} to {end_prompt}"

        fix_seed(p) # use the specified seed or get a random one if needed
        p.seed = p.n_iter*p.batch_size * [int(p.seed)] # force the seed to stay the same for each iteration

        p.extra_generation_params = {"Start percent":start_percent,"End percent":end_percent}

        processed = process_images(p)


        # gif docs: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
        if make_gif:

            n_img = len(processed.images)-1

            if n_img >= 2: 

                # note that durations are in milliseconds, and we want to linger on the first and last frames
                durations = [100] * n_img 
                durations[0] = 1000
                durations[-1] = 1000

                result_gif = processed.images[1].save('gif_test.gif', save_all=True, append_images=processed.images[2:], duration=durations, loop=0, subrectangles=True)            #for img in processed.images:
                process_images.append(result_gif)
                #    print(f"got image {img}")

        return processed


def slerp(start, end, weight):
    start_norm = start/torch.norm(start, dim=1, keepdim=True)
    end_norm = end/torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((start_norm*end_norm).sum(1))
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0-weight)*omega)/sin_omega).unsqueeze(1)*start + (torch.sin(weight*omega)/sin_omega).unsqueeze(1)*end
