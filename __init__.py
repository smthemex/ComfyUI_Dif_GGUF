# ComfyUI_Dif_GGUF - GGUF model loader for ComfyUI
# Reads GGUF files, preserves quantization, injects weights bypassing shape checks.

from .dif_gguf_node import Dif_GGUF_Loader, DifGGUFExtension
from .ggml_ops import GGMLTensor, GGMLOps, GGMLLayer, GGUFModelPatcher

NODE_CLASS_MAPPINGS = {
    "Dif_GGUF_Loader": Dif_GGUF_Loader,
}

__all__ = [
    "Dif_GGUF_Loader",
    "DifGGUFExtension",
    "GGMLTensor",
    "GGMLOps",
    "GGMLLayer",
    "GGUFModelPatcher",
]