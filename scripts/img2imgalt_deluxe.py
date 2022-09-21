from collections import namedtuple

import numpy as np
from tqdm import trange

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
        original_prompt = gr.Textbox(label="Original prompt", lines=1)
        original_negative_prompt = gr.Textbox(label="Original negative prompt", lines=1)

        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=50)
        randomness = gr.Slider(label="Randomness", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        decode_cfg = gr.Slider(label="Override Decode CFG scale", minimum=-5.0, maximum=5.0, step=0.1, value=-0.5)
        infer_cfg = gr.Slider(label="Override Infer CFG scale", minimum=-5.0, maximum=5.0, step=0.1, value=1.0)

        batch_mode = gr.Dropdown(label="Batch mode", choices=["Decode","Generate","Decode and Generate"], value="Decode")

        input_dir = gr.Textbox(label="Input image directory", lines=1)
        input_dir = gr.Textbox(label="Output noise directory", lines=1)
        output_dir = gr.Textbox(label="Output image directory", lines=1)

        return [original_prompt, original_negative_prompt, st, randomness,decode_cfg,infer_cfg, input_dir, output_dir, batch_mode]

    def run(self, p, original_prompt, original_negative_prompt, st, randomness,decode_cfg,infer_cfg, input_dir, output_dir, batch_mode):
        p.batch_size = 1
        p.batch_count = 1

        p.cfg_scale = infer_cfg

        def sample_extra(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
            lat = (p.init_latent.cpu().numpy() * 10).astype(int)

            shared.state.job_count += 1
            cond = p.sd_model.get_learned_conditioning(p.batch_size * [original_prompt])
            uncond = p.sd_model.get_learned_conditioning(p.batch_size * [original_negative_prompt])
            rec_noise = find_noise_for_image(p, cond, uncond, decode_cfg, st)

            rand_noise = processing.create_random_tensors(p.init_latent.shape[1:], [p.seed + x + 1 for x in range(p.init_latent.shape[0])])
            
            combined_noise = ((1 - randomness) * rec_noise + randomness * rand_noise) / ((randomness**2 + (1-randomness)**2) ** 0.5)
            
            sampler = samplers[p.sampler_index].constructor(p.sd_model)

            sigmas = sampler.model_wrap.get_sigmas(p.steps)
            
            noise_dt = combined_noise - (p.init_latent / sigmas[0])
            
            p.seed = p.seed + 1
            
            return sampler.sample_img2img(p, p.init_latent, noise_dt, conditioning, unconditional_conditioning)

        p.sample = sample_extra

        processed = processing.process_images(p)

        return processed

