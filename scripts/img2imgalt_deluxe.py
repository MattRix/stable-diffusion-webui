from collections import namedtuple

import numpy as np
from tqdm import trange

import os
import math
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, prompt_parser
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state

import torch
import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat


def find_noise_for_image(p, cond, uncond, cfg_scale, steps):
    x = p.init_latent

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(shared.sd_model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    shared.state.sampling_steps = steps

    for i in trange(1, len(sigmas)):
        shared.state.sampling_step += 1

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
        t = dnw.sigma_to_t(sigma_in)

        eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * -cfg_scale

        d = (x - denoised) / sigmas[i]
        dt = sigmas[i] - sigmas[i - 1]

        x = x + d * dt

        sd_samplers.store_latent(x)

        # This shouldn't be necessary, but solved some VRAM issues
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt

    shared.state.nextjob()

    return x / x.std()


class Script(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "img2img alt deluxe"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        original_prompt = gr.Textbox(label="Original prompt", lines=1, value="People staring")
        original_negative_prompt = gr.Textbox(label="Original negative prompt", lines=1)

        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=50)

        decode_cfg = gr.Slider(label="Override Decode CFG scale", minimum=-5.0, maximum=5.0, step=0.1, value=-0.7)
        infer_cfg = gr.Slider(label="Override Infer CFG scale", minimum=-5.0, maximum=5.0, step=0.1, value=1.2)

        batch_mode = gr.Dropdown(label="Batch mode", choices=["Decode","Generate","Decode and Generate"], value="Decode")


        with gr.Row():
            with gr.Column():
                in_images_dir = gr.Textbox(label="Input image directory", lines=1, value="anims/jp/in_images")
                out_noise_dir = gr.Textbox(label="Output noise directory", lines=1, value="anims/jp/out_noise")
                out_images_dir = gr.Textbox(label="Output image directory", lines=1, value="anims/jp/out_images")
                
            with gr.Column():
                max_images = gr.Number(label="Max image count", value=0)
                first_image_index = gr.Number(label="First image index", value=0)

        return [original_prompt, original_negative_prompt, st,decode_cfg,infer_cfg, in_images_dir, out_noise_dir, out_images_dir, max_images,first_image_index, batch_mode]

    def run(self, p, original_prompt, original_negative_prompt, st,decode_cfg,infer_cfg, in_images_dir, out_noise_dir, out_images_dir, max_images, first_image_index, batch_mode):

        print(f"input path is {in_images_dir}")
        image_paths = [file for file in [os.path.join(in_images_dir, x) for x in os.listdir(in_images_dir)] if os.path.isfile(file)]

        if first_image_index < 0: 
            first_image_index = 0

        if max_images <= 0:
            max_images = len(image_paths)

        #allow processing only a few images, useful for testing etc 
        
        image_paths = image_paths[int(first_image_index):int(max_images)]

        images = []

        for path in image_paths:
            print(f"checking path {path}")
            try:
                img = Image.open(path)
                images.append((img, path))
            except:
                print(f"Error processing {path}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        p.batch_count = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        p.cfg_scale = infer_cfg

        self.write_noise = True
        self.write_images = True

        #todo: we have to init the model somehow... we can't do this stuff without an initialized model 
        #do it in sample! we do everything in sample
        #we should have a variable called "max images" or something where we only do that many images (or do all if blank/0/-1)

        print(f"we are passing {len(p.init_images)} vs {len(images)}")

        self.iterator = 0

        #TODO: process each image one by one (you can later also do these in a batch but not for now)

        def sample_extra(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
            
            print(f"sampling image for iterator {self.iterator} at {self.image_path}")

            self.iterator += 1    

            shared.state.job_count += 1
            cond = p.sd_model.get_learned_conditioning(p.batch_size * [original_prompt])
            uncond = p.sd_model.get_learned_conditioning(p.batch_size * [original_negative_prompt])
            noise = find_noise_for_image(p, cond, uncond, decode_cfg, st)
            
            if self.write_noise:
                noisepath = os.path.basename(self.image_path)
                noisepath = os.path.join(out_noise_dir, noisepath)
                noisepath = noisepath+".pt"
                #np.savetxt(outpath, noise.numpy())
                torch.save(noise,noisepath)

            if self.write_images:
                sampler = samplers[p.sampler_index].constructor(p.sd_model)

                sigmas = sampler.model_wrap.get_sigmas(p.steps)
                
                noise_dt = noise - (p.init_latent / sigmas[0])
                
                p.seed = p.seed + 1
                
                return sampler.sample_img2img(p, p.init_latent, noise_dt, conditioning, unconditional_conditioning)
            else:
                return None


        fullproc = Processed(p, [], p.seed, "")



        p.sample = sample_extra


        state.job_count = len(images)

        for i,(img,path) in enumerate(images):
            p.init_images = [img]
            self.image = img
            self.image_path = path

            state.job = f"{i+1} out of {len(images)}: {self.image_path}"

            proc = processing.process_images(p)
            result_image = proc.images[0]
            fullproc.images.append(result_image)
            
            if self.write_images:
                filename = os.path.basename(path)
                result_image.save(os.path.join(out_images_dir, filename))


        #processed.images = []

        #return Processed(p, [], p.seed, "")

        return fullproc

