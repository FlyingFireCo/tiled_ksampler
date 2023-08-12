
import comfy.diffusers_convert
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.clip_vision
import torch
import nodes

class Tiled_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"model": ("MODEL", ),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "tiling": ("INT", {"default": 1, "min": 0, "max": 1}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Sampling/Tiled"
    def apply_circular(self, model, enable):
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def sample(self, model, seed, tiling, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        self.apply_circular(model.model, tiling == 1)
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)


NODE_CLASS_MAPPINGS = {
        "Tiled KSampler": Tiled_KSampler,
}