
import os
import gc
import torch
import warnings
import folder_paths
import logging
from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension

import comfy
import comfy.utils
import comfy.sd
import comfy.model_management
import comfy.model_detection
import comfy.model_patcher

# update_folder_names_and_paths from @city96 https://github.com/city96/ComfyUI-GGUF
def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

# add gguf folder to comfy optional
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path)


#  Some codes from @city96 https://github.com/city96/ComfyUI-GGUF  and  used for diffusion_gguf model
import gguf as gguf_lib
from .ggml_ops import (GGMLTensor, GGMLOps,is_quantized, GGUFModelPatcher)
from .tokenizer_map import convert_text_encoder_state_dict


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != gguf_lib.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf_lib.GGUFValueType.ARRAY or field.types[1] != gguf_lib.GGUFValueType.INT32:
        raise TypeError(f"Bad orig shape metadata for {field_key}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def read_gguf(path):
    """Read GGUF → dict (keeps original key names)."""
    reader = gguf_lib.GGUFReader(path)
    sd = {}
    qtypes = {}
    for t in reader.tensors:
        k = t.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(t.data)
        shape = get_orig_shape(reader, t.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(t.shape)))
        if t.tensor_type in {gguf_lib.GGMLQuantizationType.F32, gguf_lib.GGMLQuantizationType.F16}:
            sd[k] = raw.view(*shape)
        else:
            sd[k] = GGMLTensor(raw, tensor_type=t.tensor_type, tensor_shape=shape)
        qn = getattr(t.tensor_type, "name", repr(t.tensor_type))
        qtypes[qn] = qtypes.get(qn, 0) + 1
    del reader
    logging.info("GGUF: " + ", ".join(f"{q}×{n}" for q, n in qtypes.items()))
    qsd = {k: v for k, v in sd.items() if is_quantized(v)}
    if qsd:
        max_key = max(qsd, key=lambda k: qsd[k].numel())
        sd[max_key].is_largest_weight = True
    return sd


def safe_state_dict_load(model, state_dict, assign):
    success = 0
    fail = 0
    for k, v in state_dict.items():
        if k in model._parameters:
            existing = model._parameters[k]
            try:
                # Reference-mount quantized tensors instead of copying them.
                #
                # existing.data.copy_(v) writes into the full-size bf16 tensor
                # that model_config.get_model() pre-allocated, which forces a
                # dequantization of every GGUF tensor at load time and keeps the
                # whole checkpoint resident in RAM at full precision -- the Q4
                # size advantage is lost entirely.
                #
                # set_attr_param() wraps v in nn.Parameter without copying, so
                # the GGMLTensor subclass (and its quantized storage) survives.
                # GGMLLayer.cast_bias_weight() then dequantizes per-layer on
                # demand at inference time, which is the intended design.
                # The pre-allocated bf16 tensor loses its last reference here
                # and is reclaimed by GC.
                if is_quantized(v) or (assign and existing.is_meta):
                    comfy.utils.set_attr_param(model, k, v)
                else:
                    existing.data.copy_(v)
                success += 1
            except Exception as e:
                logging.warning(f"  Skipped param '{k}': {e}")
                fail += 1
        elif k in model._buffers:
            existing = model._buffers[k]
            if existing.shape == v.shape:
                try:
                    existing.copy_(v)
                    success += 1
                except Exception as e:
                    logging.warning(f"  Skipped buffer '{k}': {e}")
                    fail += 1
            else:
                try:
                    comfy.utils.set_attr_buffer(model, k, v)
                except Exception as e:
                    logging.warning(f"  Skipped buffer (replace) '{k}': {e}")
                fail += 1
        else:
            parts = k.rsplit(".", 1)
            if len(parts) == 2:
                obj_path, attr = parts
                obj = model
                try:
                    for p in obj_path.split("."):
                        obj = getattr(obj, p)
                except AttributeError:
                    fail += 1
                    continue
                if hasattr(obj, attr):
                    existing = getattr(obj, attr)
                    try:
                        if isinstance(existing, torch.nn.Parameter):
                            comfy.utils.set_attr_param(obj, attr, v)
                        else:
                            comfy.utils.set_attr_buffer(obj, attr, v)
                        success += 1
                    except Exception as e:
                        logging.warning(f"  Skipped nested '{k}': {e}")
                        fail += 1
                else:
                    fail += 1
            else:
                fail += 1

    if fail > 0:
        logging.warning(f"Safe load: {success} OK, {fail} skipped")

    for k, v in state_dict.items():
        # Quantized tensors were reference-mounted above and legitimately have
        # a packed shape that differs from the unquantized parameter shape.
        # Do NOT "fix" them here: .to(dtype=float16) would dequantize and
        # materialise the full-precision tensor, reintroducing the RAM blowup.
        if is_quantized(v):
            continue
        if k in model._parameters:
            existing = model._parameters[k]
            if existing.shape != v.shape:
                try:
                    v_clean = v.to(existing.device, dtype=torch.float16)
                    comfy.utils.set_attr_param(model, k, v_clean)
                    success += 1
                    logging.warning(f"  Replaced '{k}': {list(existing.shape)} → {list(v_clean.shape)}")
                except Exception as e:
                    logging.warning(f"  Replace failed '{k}': {e}")
        elif k in model._buffers:
            existing = model._buffers[k]
            if existing.shape != v.shape:
                try:
                    v_clean = v.to(existing.device, dtype=torch.float16)
                    comfy.utils.set_attr_buffer(model, k, v_clean)
                    success += 1
                    logging.warning(f"  Replaced buffer '{k}': {list(existing.shape)} → {list(v_clean.shape)}")
                except Exception as e:
                    logging.warning(f"  Replace buffer failed '{k}': {e}")
    return


def load_gguf_model(sd, ops=None):
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    model_config = comfy.model_detection.model_config_from_unet(sd, "")
    if model_config is None:
        new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = comfy.model_detection.model_config_from_unet(new_sd, "")
            if model_config is not None:
                sd = new_sd
    if model_config is None:
        model_config = comfy.model_detection.model_config_from_diffusers_unet(sd)
        if model_config is not None:
            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
            sd.update(new_sd)
    if model_config is None:
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None
    unet_dtype = comfy.model_management.unet_dtype(
        model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    if model_config.quant_config:
        manual_cast = comfy.model_management.unet_manual_cast(None, load_device, unet_weight_dtype)
    else:
        manual_cast = comfy.model_management.unet_manual_cast(unet_dtype, load_device, unet_weight_dtype)
    model_config.set_inference_dtype(unet_dtype, manual_cast)
    if ops is not None:
        model_config.custom_operations = ops

    inital_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
    model = model_config.get_model(sd, "", device=inital_device)

    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)

    target = model.diffusion_model
    safe_state_dict_load(target, sd, assign=patcher.is_dynamic())

    # The bf16 tensors pre-allocated by get_model() were replaced by
    # reference-mounted GGMLTensors and are now unreferenced. Collect so the
    # freed memory is actually returned rather than lingering in the allocator.
    # (The caller drops its own `sd` reference right after this returns.)
    gc.collect()

    return patcher

class Dif_GGUF_Loader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):

        return IO.Schema(
            node_id="Dif_GGUF_Loader",
            display_name="Dif_GGUF_Loader",
            description="Load gguf model (quantized lazy dequant + shape-tolerant)",
            category="model/loaders",
            inputs=[
                IO.Combo.Input("gguf", options=["none"] + folder_paths.get_filename_list("gguf")+folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[IO.Model.Output(display_name="model"),]
        )

    @classmethod
    def execute(cls, gguf) -> IO.NodeOutput:
        gguf_path = folder_paths.get_full_path("gguf", gguf) if gguf != "none" and  gguf  in ["none"] + folder_paths.get_filename_list("gguf") else  folder_paths.get_full_path("diffusion_models", gguf) if  gguf != "none" and  gguf  in ["none"] + folder_paths.get_filename_list("diffusion_models")  else None
        assert gguf_path is not None
        if not gguf_path.endswith(".gguf"):
            model_options={}
            if "fp8_e4m3fn" in gguf_path:
                model_options["dtype"] = torch.float8_e4m3fnune
            elif "fp8_e4m3fn_fast" in gguf_path:
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif  "fp8_e5m2" in gguf_path:
                model_options["dtype"] = torch.float8_e5m2
            model = comfy.sd.load_diffusion_model(gguf_path, model_options=model_options)
            logging.warning(f"You choice {gguf_path} is not gguf, back to load as diffusion model: {type(model).__name__}")
            return IO.NodeOutput(model)

        sd = read_gguf(gguf_path)
        logging.info(f"GGUF: {len(sd)} tensors")    

        has_quant = any(is_quantized(v) for v in sd.values())
        ops = GGMLOps() if has_quant else None

        model = load_gguf_model(sd, ops)
        del sd
        if model is None:
            raise RuntimeError("Could not detect model type from GGUF")

        if has_quant:
            model = GGUFModelPatcher.clone(model)
            model.size = 0

        logging.info(f"Loaded: {type(model.model).__name__}")
        return IO.NodeOutput(model)


# ── Text encoder (CLIP) GGUF ────────────────────────────────────────────
# Mirrors the unet path above, but hands the state dict to ComfyUI's
# load_text_encoder_state_dicts() with custom_operations=GGMLOps so the
# quantized tensors stay quantized and are dequantized lazily at forward time.
# No ComfyUI source file is modified: model_options is a public entry point.

CLIP_GGUF_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace",
    "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image",
    "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage",
    "mage", "minimax",
]

CLIP_GGUF_DUAL_TYPES = [
    "sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image",
    "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace",
]


def _clip_gguf_full_path(name):
    """Resolve a text encoder file from either the gguf or text_encoders folder."""
    for folder in ("clip_gguf", "gguf", "text_encoders"):
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:
            path = None
        if path is not None:
            return path
    return None


def read_text_encoder_gguf(path):
    """Read a text encoder GGUF and normalize its key names."""
    sd = read_gguf(path)
    sd, arch = convert_text_encoder_state_dict(sd)
    logging.info(f"GGUF TE: {len(sd)} tensors (arch={arch})")
    return sd


def load_text_encoder_gguf(paths, clip_type_name, device="default"):
    """Load one or more text encoders, any of which may be a .gguf file."""
    clip_type = getattr(comfy.sd.CLIPType, clip_type_name.upper(),
                        comfy.sd.CLIPType.STABLE_DIFFUSION)

    state_dicts = []
    has_quant = False
    for path in paths:
        if path.endswith(".gguf"):
            sd = read_text_encoder_gguf(path)
            if any(is_quantized(v) for v in sd.values()):
                has_quant = True
        else:
            sd = comfy.utils.load_torch_file(path, safe_load=True)
        state_dicts.append(sd)

    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    if has_quant:
        # Public hook: sd1_clip reads model_options["custom_operations"].
        model_options["custom_operations"] = GGMLOps
        # GGUF norm tensors are F32, so ComfyUI's t5xxl_detect/llama_detect would
        # pick float32 as the compute dtype. Force a sane one instead.
        compute_dtype = comfy.model_management.text_encoder_dtype(
            comfy.model_management.text_encoder_device()
        )
        model_options["dtype_t5"] = compute_dtype
        model_options["dtype_llama"] = compute_dtype

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )

    if has_quant:
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        clip.patcher.size = 0
    return clip


class CLIP_GGUF_Loader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = ["none"] + folder_paths.get_filename_list("clip_gguf") \
                  + folder_paths.get_filename_list("gguf")
        return IO.Schema(
            node_id="CLIP_GGUF_Loader",
            display_name="CLIP_GGUF_Loader",
            description="Load a text encoder from GGUF (quantized lazy dequant)",
            category="model/loaders",
            inputs=[
                IO.Combo.Input("gguf", options=options),
                IO.Combo.Input("type", options=CLIP_GGUF_TYPES),
                IO.Combo.Input("device", options=["default", "cpu"], optional=True),
            ],
            outputs=[IO.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def execute(cls, gguf, type="stable_diffusion", device="default") -> IO.NodeOutput:
        path = _clip_gguf_full_path(gguf) if gguf != "none" else None
        assert path is not None, f"Text encoder not found: {gguf}"
        clip = load_text_encoder_gguf([path], type, device)
        logging.info(f"Loaded text encoder: {type}")
        return IO.NodeOutput(clip)


class DualCLIP_GGUF_Loader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = ["none"] + folder_paths.get_filename_list("clip_gguf") \
                  + folder_paths.get_filename_list("gguf")
        return IO.Schema(
            node_id="DualCLIP_GGUF_Loader",
            display_name="DualCLIP_GGUF_Loader",
            description="Load two text encoders, either of which may be GGUF",
            category="model/loaders",
            inputs=[
                IO.Combo.Input("gguf1", options=options),
                IO.Combo.Input("gguf2", options=options),
                IO.Combo.Input("type", options=CLIP_GGUF_DUAL_TYPES),
                IO.Combo.Input("device", options=["default", "cpu"], optional=True),
            ],
            outputs=[IO.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def execute(cls, gguf1, gguf2, type="flux", device="default") -> IO.NodeOutput:
        path1 = _clip_gguf_full_path(gguf1) if gguf1 != "none" else None
        path2 = _clip_gguf_full_path(gguf2) if gguf2 != "none" else None
        assert path1 is not None, f"Text encoder not found: {gguf1}"
        assert path2 is not None, f"Text encoder not found: {gguf2}"
        clip = load_text_encoder_gguf([path1, path2], type, device)
        logging.info(f"Loaded dual text encoder: {type}")
        return IO.NodeOutput(clip)


class DifGGUFExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Dif_GGUF_Loader, CLIP_GGUF_Loader, DualCLIP_GGUF_Loader]

async def comfy_entrypoint() -> DifGGUFExtension:
    return DifGGUFExtension()