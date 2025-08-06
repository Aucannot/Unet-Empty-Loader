import torch
import comfy
import comfy.model_management
import comfy.sd
import folder_paths

class UNETEmptyLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_empty"
    CATEGORY = "advanced/loaders"

    def load_unet_empty(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        sd = comfy.utils.load_torch_file(unet_path)
        dtype = model_options.get("dtype", None)
        diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd)
        load_device = comfy.model_management.get_torch_device()
        model_config = comfy.sd.model_detection.model_config_from_unet(sd, "")
        if model_config is not None:
            new_sd = sd
        else:
            new_sd = comfy.sd.model_detection.convert_diffusers_mmdit(sd, "")
            if new_sd is not None:
                model_config = comfy.sd.model_detection.model_config_from_unet(new_sd, "")
                if model_config is None:
                    raise RuntimeError("ERROR: Could not detect model type.")
            else:
                model_config = comfy.sd.model_detection.model_config_from_diffusers_unet(sd)
                if model_config is None:
                    raise RuntimeError("ERROR: Could not detect model type.")
                diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
                new_sd = {}
                for k in diffusers_keys:
                    if k in sd:
                        new_sd[diffusers_keys[k]] = sd.pop(k)
        offload_device = comfy.model_management.unet_offload_device()
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.scaled_fp8 is not None:
            weight_dtype = None
        if dtype is None:
            unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
        else:
            unet_dtype = dtype
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
        if model_options.get("fp8_optimizations", False):
            model_config.optimizations["fp8"] = True
        model = model_config.get_model(new_sd, "")
        model = model.to(offload_device)
        # without load weights from safetensors
        return (comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device),)

NODE_CLASS_MAPPINGS = {
    "UNETEmptyLoader": UNETEmptyLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UNETEmptyLoader": "UNet Empty Loader",
}