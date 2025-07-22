import torch
import comfy.model_management
import logging
from comfy.model_patcher import ModelPatcher

from comfy.ldm.flux.model import Flux as FluxTransformer
from comfy.model_base import Flux as FluxBase
from comfy.supported_models import Flux as FluxConfig


flux_params = {
    "in_channels": 16,
    "out_channels": 16,
    "vec_in_dim": 768,
    "context_in_dim": 4096,
    "hidden_size": 3072,
    "mlp_ratio": 4.0,
    "num_heads": 24,
    "depth": 19,
    "depth_single_blocks": 38,
    "axes_dim": [16, 56, 56],
    "theta": 10_000,
    "patch_size": 2,
    "qkv_bias": True,
    "guidance_embed": True,
}


class FluxEmptyLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (
                    ["flux-dev", "flux-schnell", "flux-kontext"],
                    {
                        "default": "flux-dev",
                    },
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                    },
                ),
            },
            "optional": {
                "device": (
                    ["cpu", "cuda"],
                    {
                        "default": "cuda",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_flux_empty"
    CATEGORY = "loaders/flux"
    DESCRIPTION = "Empty Flux model loader for hijacking forward"

    def load_flux_empty(self, model_type, device="auto"):
        logging.info(f"Creating empty Flux model: {model_type}")

        # Handle device parameter properly
        if device == "cuda":
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif device == "cpu":
            device = torch.device("cpu")
        elif device == "auto":
            device = comfy.model_management.get_torch_device()
        else:
            device = torch.device(device)

        unet_config = {
            "image_model": "flux",
            "guidance_embed": True,
            "disable_unet_model_creation": True,
        }
        unet_config.update(flux_params)
        model_config = FluxConfig(unet_config)

        flux_transformer = FluxTransformer(
            **flux_params, device=device, dtype=torch.bfloat16
        )
        empty_flux = FluxBase(model_config=model_config, device=device)
        empty_flux.diffusion_model = flux_transformer
        model = ModelPatcher(
            model=empty_flux,
            load_device=device,
            offload_device=comfy.model_management.unet_offload_device(),
            size=0,
        )

        return (model,)


NODE_CLASS_MAPPINGS = {
    "FluxEmptyLoader": FluxEmptyLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxEmptyLoader": "Flux Empty Loader",
}
