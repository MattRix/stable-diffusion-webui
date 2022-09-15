from pickletools import optimize
from re import I
import imageio
from modules.images import apply_filename_pattern
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

        notes = gr.HTML(value="<div style='margin:15px'>Note: gifs will output into your grids folder. Make sure you have grids enabled if you have gifs enabled.</div>")

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

        # set up the start and end seed by pretending they're the real seed 
        # this generates a random seed if they are set to -1 etc

        orig_seed = p.seed

        p.seed = start_seed
        fix_seed(p)
        start_seed = p.seed

        p.seed = end_seed
        fix_seed(p)
        end_seed = p.seed

        p.seed = orig_seed

        start_noise = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], [start_seed])
        end_noise = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], [end_seed])

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

        p.extra_generation_params = {"Start percent":start_percent,"End percent":end_percent,"Start seed":int(start_seed),"End seed":int(end_seed)}

        processed = process_images(p)


        # gif docs: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
        if make_gif:

            n_img = len(processed.images)-1

            if n_img >= 2: 

                # note that durations are in milliseconds, and we want to linger on the first and last frames
                durations = [100] * n_img 
                durations[0] = 1000
                durations[-1] = 1000

                processed.images[1].save(get_next_gif_filename(p), save_all=True, append_images=processed.images[2:], duration=durations, loop=0, subrectangles=True)            #for img in processed.images:


        return processed


def slerp(start, end, weight):
    start_norm = start/torch.norm(start, dim=1, keepdim=True)
    end_norm = end/torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((start_norm*end_norm).sum(1))
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0-weight)*omega)/sin_omega).unsqueeze(1)*start + (torch.sin(weight*omega)/sin_omega).unsqueeze(1)*end

# this is a bit of a hack but I didn't want to recreate all the existing filename code
# and unfortunately I also couldn't just extra the filenames from that stuff
# so I had to recreate it all in a minimal way. The other option would be to keep gifs with unique file names
# but then they wouldn't match with their corresponding grids... maybe not a big deal?
def get_next_gif_filename(p):
    import os

    #find the next grid file name and use the same thing but make it a gif
    path = p.outpath_grids
    filecount = len([x for x in os.listdir(path) if os.path.splitext(x)[1] == '.png'])

    filename = ""
    for i in range(500): #incrementing from filecount, find the first grid-####.png file that doesn't exist
        filename = f"{path}/grid-{(filecount+i):04}.png"
        if not os.path.exists(filename):
            #use the file before that, since the existing one is the newly created grid
            filename = f"{path}/grid-{(filecount+i-1):04}.png"
            break

    filename = filename.replace(".png",".gif")
    filename = filename.replace("grid-","gif-")

    return filename