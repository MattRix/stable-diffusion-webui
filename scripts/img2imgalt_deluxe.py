'''
TODO: 
    - figure out how to make the min and max actually update because we need more than 5 CFG! (at worst we can just rename it, I guess)
        - turns out you can edit it in the html, which is silly but yeah 
    DONE: support outputting a null image from the thing (it can be any image, because we will make sure to not save it anyway)
    - add keyframe list
        and toggle for "should use keyframes"

notes:
- if you mix a negative prompt with negative cfg you get some cool stuff 
- small differences in denoise can make a huge difference
- play with CFG values as well
- sometimes stuff looks better at lower steps, don't be afraid of 20 or even 15 

'''

from collections import namedtuple

import numpy as np
from sympy import re
from tqdm import trange

import os
import math
import sys
import traceback
from modules import devices
import torchvision.transforms.functional as TVF

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
        original_prompt = gr.Textbox(label="Original prompt", lines=1, value="still from the film jurassic park")
        original_negative_prompt = gr.Textbox(label="Original negative prompt", lines=1)

        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=50)

        decode_cfg = gr.Slider(label="Override Decode CFG scale", minimum=-5.0, maximum=5.0, step=0.1, value=-0.7)
        infer_cfg = gr.Slider(label="Override Infer CFG scale", minimum=-10.0, maximum=15.0, step=0.1, value=1.2)
        sharpness = gr.Slider(label="Sharpness", minimum=-10.0, maximum=10.0, step=0.1, value=1.0)

        with gr.Row():
            
            with gr.Column():
                in_images_dir = gr.Textbox(label="Input image directory", lines=1, value="anims/jp/in_images")
                out_noise_dir = gr.Textbox(label="Output noise directory", lines=1, value="anims/jp/out_noise")
                out_images_dir = gr.Textbox(label="Output image directory", lines=1, value="anims/jp/out_images")
                
            with gr.Column():

                with gr.Row():
                    should_write_noise = gr.Checkbox(label="Decode Noise", value=True)
                    should_write_images = gr.Checkbox(label="Generate Images", value=True)

                first_image_index = gr.Number(label="First image index", value=0)
                max_images = gr.Number(label="Max image count", value=0)

                with gr.Row():
                    keyframe_str = gr.Textbox(label="Keyframes", lines=1, value="")
                    should_use_keyframes = gr.Checkbox(label="Use keyframes?", value=False)

        return [original_prompt, original_negative_prompt, st,decode_cfg,infer_cfg, in_images_dir, out_noise_dir, out_images_dir, max_images,first_image_index, should_write_noise, should_write_images,keyframe_str,should_use_keyframes,sharpness]

    def run(self, p, original_prompt, original_negative_prompt, st,decode_cfg,infer_cfg, in_images_dir, out_noise_dir, out_images_dir, max_images, first_image_index, should_write_noise, should_write_images,keyframe_str,should_use_keyframes,sharpness):

        self.write_noise = should_write_noise
        self.write_images = should_write_images

        os.makedirs(out_noise_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)


        image_paths = [file for file in [os.path.join(in_images_dir, x) for x in os.listdir(in_images_dir)] if os.path.isfile(file)]

        #you can specify comma delimited frame indices to use (including "-1" for the last frame)
        if should_use_keyframes:
            keyframes = [int(x) for x in keyframe_str.split(",")]
            keyframe_image_paths = [image_paths[k] for k in filter(lambda k:k<len(image_paths),keyframes)]
            image_paths = keyframe_image_paths
        else:
            if first_image_index < 0: 
                first_image_index = 0

            if max_images <= 0:
                max_images = len(image_paths)

            #allow processing only a few images, useful for testing etc 
            
            image_paths = image_paths[int(first_image_index):int(first_image_index)+int(max_images)]

        images = []

        for path in image_paths:
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

        

        #todo: we have to init the model somehow... we can't do this stuff without an initialized model 
        #do it in sample! we do everything in sample
        #we should have a variable called "max images" or something where we only do that many images (or do all if blank/0/-1)

        def sample_extra(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
   
            noisepath = os.path.basename(self.image_path)
            noisepath = os.path.join(out_noise_dir, noisepath)
            noisepath = noisepath+".pt"

            if self.write_noise:
                shared.state.job_count += 1
                cond = p.sd_model.get_learned_conditioning(p.batch_size * [original_prompt])
                uncond = p.sd_model.get_learned_conditioning(p.batch_size * [original_negative_prompt])
                noise = find_noise_for_image(p, cond, uncond, decode_cfg, st)
                torch.save(noise,noisepath)
            else:
                noise = torch.load(noisepath)

            if self.write_images:
                sampler = samplers[p.sampler_index].constructor(p.sd_model)

                sigmas = sampler.model_wrap.get_sigmas(p.steps)
                
                noise_dt = noise - (p.init_latent / sigmas[0])
                
                p.seed = p.seed + 1

                out_samples = sampler.sample_img2img(p, p.init_latent, noise_dt, conditioning, unconditional_conditioning)
                
                #1,4,48,80
                #so we have 1 row of all the images, which are each 4 (colors), then 48,80 which are w & h divided by 8
                #I'm not sure how that gets turned into images though... but we could populate it with either 1 zeroed image or the correct number 
                #print(f"got samples {out_samples.size()}")

                return out_samples
            else:
                return torch.zeros(1,4,p.height // processing.opt_f,p.width // processing.opt_f,device=devices.device) #null data


        fullproc = Processed(p, [], p.seed, "")



        p.sample = sample_extra


        state.job_count = len(images)

        for i,(img,path) in enumerate(images):
            p.init_images = [img]
            self.image = img
            self.image_path = path

            state.job = f"{i+1} out of {len(images)}: {self.image_path}"

            proc = processing.process_images(p)

            if self.write_images:
                result_image = proc.images[0]

                if sharpness != 1.0: 
                    result_image = TVF.adjust_sharpness(result_image,sharpness)

                fullproc.images.append(result_image)
                
                filename = os.path.basename(path)
                result_image.save(os.path.join(out_images_dir, filename))


        #processed.images = []

        #return Processed(p, [], p.seed, "")

        return fullproc

