# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Adapted from ComfyUI-GGUF
import gguf
import torch
import logging
import comfy.ops
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import uuid

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

def is_quantized(tensor):
    return getattr(tensor, "tensor_type", None) not in TORCH_COMPATIBLE_QTYPES

def is_torch_compatible(tensor):
    return getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights.
    Subclasses torch.Tensor so it can be used with nn.Parameter / load_state_dict.
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches if patches is not None else []

    def __new__(cls, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on GGMLTensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


# ── Dequantization ─────────────────────────────────────────────────────
def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    """Dequantize a GGMLTensor back to a regular torch.Tensor."""
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        logging.warning(f"Falling back to numpy dequant for qtype: {qtype}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)

def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

QK_K = 256
K_SCALE_SIZE = 12

# Dequant kernels (same as City96's dequant.py)
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))
    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))

dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}


# ── GGMLLayer – custom nn.Module that holds GGMLTensor weights ──────────
def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item


class GGMLLayer(torch.nn.Module):
    """
    Mixin for nn.Module subclasses that store quantized GGMLTensor weights.
    Overrides _load_from_state_dict (for load_state_dict) and
    forward_comfy_cast_weights (for comfy.ops.cast_bias_weight integration).
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = TORCH_COMPATIBLE_QTYPES

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Intercept load_state_dict to assign GGMLTensor weights directly."""
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Fix embedding shape mismatch for large vocab
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        """Override save to provide fake state dict for vram estimation."""
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

    def get_weight(self, tensor, dtype):
        """Dequantize tensor and apply patches."""
        if tensor is None:
            return None
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        """Cast/dequantize weight+bias to target device/dtype (on-demand)."""
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
        weight = self.get_weight(self.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute.
    Subclasses comfy.ops.manual_cast to integrate with ComfyUI's weight casting system.
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # Use full super().__init__() chain so disable_weight_init sets up weight/bias
            # This ensures weight is NEVER None (guarantees s.weight.device won't crash)
            super().__init__(in_features, out_features, bias, device, dtype)

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type,
                self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class RMSNorm(GGMLLayer, comfy.ops.manual_cast.RMSNorm):
        # Required by T5 / LLaMA / Gemma text encoders.
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, _bias = self.cast_bias_weight(input)
            return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)


import collections
import comfy.float


def is_ggml_pin_unsafe(module, param_name=None):
    """
    True if `module` holds mmap-backed / quantized GGML weights that must never
    be handed to cudaHostRegister().

    Two independent reasons pinning is impossible for these:
      * the pages are read-only, file-backed (mmap) -> CUDA refuses to
        page-lock them;
      * GGMLTensor.shape reports the *dequantized* logical shape, so any
        nbytes / vram_aligned_size() computation is several times larger than
        the real quantized buffer, making the registration range run past the
        end of the mapping.
    """
    if isinstance(module, GGMLLayer):
        # Fast path: a GGML layer is unsafe as soon as any of its weights is
        # still in quantized (mmap) form.
        try:
            if module.is_ggml_quantized():
                return True
        except Exception:
            pass

    names = (param_name,) if param_name else ("weight", "bias")
    for name in names:
        param = getattr(module, name, None)
        if param is None:
            continue
        # `param` may be an nn.Parameter wrapping a GGMLTensor; unwrap.
        inner = getattr(param, "data", param)
        if isinstance(inner, GGMLTensor) or is_quantized(inner):
            return True
    return False


def _install_pinned_memory_guard():
    """
    comfy has a SECOND, independent pinning path used by the dynamic/AIMDO
    weight loader:

        comfy/ops.py:226 -> comfy.pinned_memory.pin_memory(m, ...)

    It sizes the host buffer with
    `vram_aligned_size([module.weight, module.bias])`, which for a GGMLTensor
    returns the dequantized size (~4x the real Q4_K buffer). The subsequent
    cudaHostRegister() therefore always fails and logs "Pin error.", and every
    failure additionally runs discard_cuda_async_error() -> a full CUDA
    synchronize. With hundreds of layers this is both noisy and slow.

    GGUF weights are dequantized per-layer straight onto CUDA at inference
    time, so a pinned host staging buffer buys us nothing. Skip it.
    """
    try:
        import comfy.pinned_memory as _pm
    except Exception:
        return
    if getattr(_pm, "_gguf_guard_installed", False):
        return

    _orig_pin_memory = _pm.pin_memory
    _orig_get_pin = _pm.get_pin

    def pin_memory(module, subset="weights", size=None):
        if is_ggml_pin_unsafe(module):
            return
        return _orig_pin_memory(module, subset=subset, size=size)

    def get_pin(module, subset="weights"):
        if is_ggml_pin_unsafe(module):
            return None
        return _orig_get_pin(module, subset=subset)

    _pm.pin_memory = pin_memory
    _pm.get_pin = get_pin
    _pm._gguf_guard_installed = True

    # comfy/ops.py does `import comfy.pinned_memory` and calls through the
    # module attribute, so patching the module is enough. Guard anyway in case
    # a future version imports the names directly.
    try:
        import comfy.ops as _ops
        if getattr(_ops, "pin_memory", None) is _orig_pin_memory:
            _ops.pin_memory = pin_memory
        if getattr(_ops, "get_pin", None) is _orig_get_pin:
            _ops.get_pin = get_pin
    except Exception:
        pass


_install_pinned_memory_guard()


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    ModelPatcher subclass that handles patches on GGMLTensor (quantized) weights.
    Replaces default patch_weight_to_device to store patches on the GGMLTensor
    instead of applying them to the dequantized weight immediately.
    """
    patch_on_device = False
    # Module-name prefixes of block-swap-owned blocks; these are excluded from
    # mmap release so their GGUF pages stay file-backed and OS-reclaimable.
    _blockswap_prefixes = ()
    mmap_released = False
    named_modules_to_munmap = {}
    # name -> module, built once per instance by pin_weight_to_device so we
    # don't rebuild dict(named_modules()) for every single weight key.
    # Deliberately None here: a mutable class-level dict would be shared across
    # every patcher instance (clone() swaps __class__ in place).
    _gguf_module_cache = None

    # LoRA patches for *quantized* GGUF weights live here instead of self.patches.
    # comfy's lowvram path iterates self.patches to build LowVramPatch objects
    # that apply the patch onto a throwaway dequant copy. For GGUF that copy is
    # discarded by block-swap's `p.data = orig` restore, silently dropping the
    # LoRA. Keeping quantized-weight patches out of self.patches stops comfy from
    # building LowVramPatch for them, so they stay mounted on the mmap-backed
    # GGMLTensor (.patches) and survive swap restores.
    _ggml_patches = None

    @classmethod
    def clone(cls, model):
        """Convert a regular ModelPatcher to GGUFModelPatcher (like City96)."""
        src_cls = model.__class__
        model.__class__ = cls
        n = model
        n.patch_on_device = getattr(model, "patch_on_device", False)
        n.mmap_released = getattr(model, "mmap_released", False)
        if src_cls != cls:
            n.size = 0
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """Like ModelPatcher.add_patches, but quantized GGUF weights keep their
        LoRA off comfy's lowvram path.

        Quantized keys: store in self._ggml_patches and apply immediately via
        patch_weight_to_device() (which mounts the patch on the mmap GGMLTensor).
        Non-quantized keys (F32/F16): delegate to the parent implementation so
        comfy's normal LowVramPatch machinery handles them.
        """
        if self._ggml_patches is None:
            self._ggml_patches = {}
        applied = []
        # Split patches into quantized vs torch-compatible so each goes the right way.
        quant_keys = []
        torch_keys = {}
        model_sd_keys = set(self.model_state_dict().keys())
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]
            if key not in model_sd_keys:
                continue
            weight = comfy.utils.get_attr(self.model, key)
            if is_quantized(weight):
                quant_keys.append(k)
                current = self._ggml_patches.get(key, [])
                current.append((strength_patch, patches[k], strength_model, offset, function))
                self._ggml_patches[key] = current
                applied.append(k)
            else:
                torch_keys[k] = patches[k]

        # Apply quantized patches onto the mmap tensor right now.
        for k in quant_keys:
            if isinstance(k, str):
                key = k
            else:
                key = k[0]
            try:
                self.patch_weight_to_device(key)
            except Exception as e:
                logging.warning(f"GGUF LoRA apply failed for {key}: {e}")

        if torch_keys:
            applied += super().add_patches(torch_keys, strength_patch, strength_model)
        self.patches_uuid = uuid.uuid4()
        return applied

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        # Patches may come from either comfy's self.patches (non-quantized) or our
        # own self._ggml_patches (quantized, kept off the lowvram path).
        if key not in self.patches and (self._ggml_patches is None or key not in self._ggml_patches):
            return
        patches = self.patches.get(key, []) + (self._ggml_patches.get(key, []) if self._ggml_patches else [])
        weight = comfy.utils.get_attr(self.model, key)
        if is_quantized(weight):
            patches = move_patch_to_device(patches,
                self.load_device if self.patch_on_device else self.offload_device)
            # CRITICAL: mount the patch on the ORIGINAL mmap-backed GGMLTensor
            # (the `weight` we were given), not only on the dequantized copy.
            # Block-swap restores swapped blocks with `p.data = orig` where
            # `orig` is that very mmap tensor; if the patch lived only on the
            # dequant copy, the swap-restore would point the parameter back at a
            # patchless mmap tensor and the LoRA would silently stop applying.
            # GGMLTensor.to() copies `.patches` onto the dequant copy too, so
            # both the on-disk tensor and the compute copy stay LoRA-live.
            try:
                weight.patches = [(patches, key)]
            except Exception:
                pass
            out_weight = weight.to(device_to)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if patches:
                    p.patches = []
        if self._ggml_patches:
            self._ggml_patches = {}
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def _collect_blockswap_prefixes(self):
        """Collect module-name prefixes owned by a block-swap container.

        Block swappers (e.g. UniBlockSwap) replace a diffusion model's block
        container with a custom nn.ModuleList that keeps the swap blocks on the
        offload device and streams them to the compute device on demand.

        Those blocks must NOT be munmap'd: the mmap'd GGUF pages are backed by
        the model file and are reclaimable by the OS, whereas a
        .to(load_device).to(offload_device) round trip reallocates them as
        anonymous heap memory that is never reclaimed (and is subsequently
        page-locked by pinning). For a 20GB model that turns essentially the
        whole checkpoint into resident, unswappable RAM.

        Returns a tuple of name prefixes such as ("blocks.12", "blocks.13", ...).
        Empty when no block swapper is installed.
        """
        prefixes = []
        try:
            model = self.model
            for name, module in model.named_modules():
                non_swap = getattr(module, "non_swap_count", None)
                total = getattr(module, "total_count", None)
                # Duck-type the swap container rather than importing the
                # block-swap package, which may not be installed.
                if non_swap is None or total is None:
                    continue
                if not hasattr(module, "offload_swap_blocks"):
                    continue
                for idx in range(non_swap, total):
                    prefixes.append(f"{name}.{idx}" if name else str(idx))
        except Exception:
            return ()
        return tuple(prefixes)

    @staticmethod
    def _is_under_prefix(name, prefixes):
        for p in prefixes:
            if name == p or name.startswith(p + "."):
                return True
        return False

    def pin_weight_to_device(self, key):
        # GGUF quantized weights live in the GGMLTensor (mmap-backed) view.
        # They must NOT be materialised (dequantized) here: doing
        # `m.to(load_device).to(offload_device)` on a GGMLTensor fully
        # dequantizes the whole model into resident RAM -- this is exactly the
        # 14G -> 57G memory blow-up observed at sampling time.
        #
        # The weights stay file-backed (OS-reclaimable) and are lazily
        # dequantized per-layer by GGMLLayer.cast_bias_weight() onto the
        # compute device (CUDA) at inference time, then discarded.
        #
        # We also must NOT let comfy try to cudaHostRegister() them:
        #  * they are read-only, file-backed (mmap) pages, which CUDA refuses
        #    to page-lock -> "Pin error." warnings;
        #  * GGMLTensor.shape reports the *dequantized* logical shape, so
        #    `tensor.nbytes` is far larger than the real quantized buffer and
        #    the registration range would run past the mapping anyway.
        # Every failed pin also runs discard_cuda_async_error(), which does a
        # full CUDA synchronize -- hundreds of those noticeably slow loading.
        # So: skip pinning entirely for quantized GGML weights.
        if not self.mmap_released and key.rsplit('.', 1)[0] in self.named_modules_to_munmap:
            del self.named_modules_to_munmap[key.rsplit('.', 1)[0]]

        # Resolve the owning module directly instead of going through
        # get_key_weight() (which raises for keys like a missing ".bias").
        module_name, _, param_name = key.rpartition('.')
        cache = self.__dict__.get("_gguf_module_cache")
        if cache is None:
            cache = dict(self.model.named_modules())
            self._gguf_module_cache = cache
        module = cache.get(module_name)

        if module is not None and is_ggml_pin_unsafe(module, param_name):
            return  # not pinnable: mmap-backed / quantized

        super().pin_weight_to_device(key)

    def load(self, *args, force_patch_weights=False, **kwargs):
        # GGUF quantized weights are mmap-backed GGMLTensors. They must stay
        # on disk and be lazily dequantized per-layer onto the compute device
        # (CUDA) at inference time, then discarded -- never materialized into
        # resident RAM. To guarantee that, we force the lowvram/offloaded path
        # (every module goes through pin_weight_to_device, which now keeps the
        # mmap view intact) instead of the `load_completely` path whose
        # `m.to(device_to)` would dequantize layers into RAM.
        # Callers may pass `device_to` / `lowvram_model_memory` positionally
        # (e.g. ModelPatcher.partially_load -> self.load(device_to, ...)).
        # Normalize them into kwargs so we never end up passing the same
        # argument twice to super().load().
        args = list(args)
        for pos_name in ("device_to", "lowvram_model_memory"):
            if not args:
                break
            kwargs[pos_name] = args.pop(0)
        args = tuple(args)

        if "lowvram_model_memory" not in kwargs:
            # Force the lazy/offloaded path: a tiny budget (< any single
            # module) makes every module take the lowvram branch, where the
            # weight stays an mmap-backed GGMLTensor and is dequantized per
            # layer onto CUDA at inference time. This avoids the
            # `load_completely` path whose `m.to(device_to)` would dequantize
            # layers into host RAM.
            kwargs["lowvram_model_memory"] = 1
        # Make sure the one-shot `load_completely` branch, if it ever fires,
        # dequantizes onto the GPU rather than into host RAM.
        if kwargs.get("device_to") is None:
            kwargs["device_to"] = self.load_device
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())
            self._blockswap_prefixes = self._collect_blockswap_prefixes()
        super().load(force_patch_weights=force_patch_weights, *args, **kwargs)
        # NOTE: The previous mmap-release block below materialized every
        # GGMLLayer by calling `m.to(load_device).to(offload_device)`. For a
        # GGUF (quantized) model this triggers a full dequantization of ALL
        # weights into resident bf16/fp16, turning the 24G mmap-backed model
        # into ~43G+ of anonymous RAM -- exactly the 14G->57G blowup seen at
        # sampling time. GGUF weights are meant to stay mmap-backed and be
        # lazily dequantized per-layer by GGMLLayer.cast_bias_weight() at
        # inference time, so we must NOT materialize them here.
        #
        # The block-swap path (UniBlockSwap) is handled separately by keeping
        # its tensors mmap-backed (see _collect_blockswap_prefixes /
        # pin_weight_to_device), which is safe because those pages stay
        # file-backed and OS-reclaimable.
        if not self.mmap_released:
            swap_prefixes = getattr(self, "_blockswap_prefixes", ())
            skipped = sum(
                1 for n, m in self.named_modules_to_munmap.items()
                if self._is_under_prefix(n, swap_prefixes)
            )
            if skipped:
                logging.info(f"Keeping mmap for {skipped} block-swap tensors")
            self.mmap_released = True
            self.named_modules_to_munmap = {}
            self._blockswap_prefixes = ()

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        n._blockswap_prefixes = getattr(self, "_blockswap_prefixes", ())
        # Carry over any quantized-weight LoRA patches so a clone (used by the
        # LoraLoader node) keeps applying them.
        n._ggml_patches = (getattr(self, "_ggml_patches", None) or {}).copy()
        if src_cls != GGUFModelPatcher:
            n.size = 0
        return n
