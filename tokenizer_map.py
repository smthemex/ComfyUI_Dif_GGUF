# Key remapping for text-encoder GGUF files.
#
# Unet GGUF files (city96 style) keep the original ComfyUI tensor names, so the
# unet loader can consume them directly. Text-encoder GGUF files produced by
# llama.cpp's convert scripts instead use llama.cpp naming ("blk.0.attn_q.weight",
# "token_embd.weight", ...). This module converts those back to the names that
# ComfyUI's T5 / llama / gemma implementations expect.
#
# Nothing here touches ComfyUI source; it only rewrites a state dict in memory.

import logging


def sd_prefix_from_state_dict(sd):
    """Guess the architecture of a text encoder state dict."""
    keys = sd.keys()
    if any(k.startswith("enc.") or k.startswith("dec.") for k in keys):
        return "t5"
    if "token_embd.weight" in sd and any(k.endswith("attn_norm.weight") for k in keys):
        return "llama"
    if any(k.startswith("text_model.") for k in keys):
        return "clip"
    if any(k.startswith("encoder.block.") for k in keys):
        return "t5_native"
    return None


def _t5_block_map(n_block, prefix="encoder"):
    """llama.cpp T5 block names -> ComfyUI T5 block names."""
    src = f"enc.blk.{n_block}"
    dst = f"{prefix}.block.{n_block}"
    return {
        f"{src}.attn_q.weight": f"{dst}.layer.0.SelfAttention.q.weight",
        f"{src}.attn_k.weight": f"{dst}.layer.0.SelfAttention.k.weight",
        f"{src}.attn_v.weight": f"{dst}.layer.0.SelfAttention.v.weight",
        f"{src}.attn_o.weight": f"{dst}.layer.0.SelfAttention.o.weight",
        f"{src}.attn_norm.weight": f"{dst}.layer.0.layer_norm.weight",
        f"{src}.attn_rel_b.weight": f"{dst}.layer.0.SelfAttention.relative_attention_bias.weight",
        f"{src}.ffn_up.weight": f"{dst}.layer.1.DenseReluDense.wi_1.weight",
        f"{src}.ffn_gate.weight": f"{dst}.layer.1.DenseReluDense.wi_0.weight",
        f"{src}.ffn_down.weight": f"{dst}.layer.1.DenseReluDense.wo.weight",
        f"{src}.ffn_norm.weight": f"{dst}.layer.1.layer_norm.weight",
    }


def _llama_block_map(n_block):
    """llama.cpp block names -> ComfyUI llama/gemma block names."""
    src = f"blk.{n_block}"
    dst = f"model.layers.{n_block}"
    return {
        f"{src}.attn_q.weight": f"{dst}.self_attn.q_proj.weight",
        f"{src}.attn_k.weight": f"{dst}.self_attn.k_proj.weight",
        f"{src}.attn_v.weight": f"{dst}.self_attn.v_proj.weight",
        f"{src}.attn_output.weight": f"{dst}.self_attn.o_proj.weight",
        f"{src}.attn_norm.weight": f"{dst}.input_layernorm.weight",
        f"{src}.ffn_up.weight": f"{dst}.mlp.up_proj.weight",
        f"{src}.ffn_gate.weight": f"{dst}.mlp.gate_proj.weight",
        f"{src}.ffn_down.weight": f"{dst}.mlp.down_proj.weight",
        f"{src}.ffn_norm.weight": f"{dst}.post_attention_layernorm.weight",
    }


def _count_blocks(sd, pattern):
    n = 0
    while True:
        if not any(k.startswith(pattern.format(n)) for k in sd):
            break
        n += 1
    return n


def convert_t5_state_dict(sd):
    """llama.cpp T5 GGUF -> ComfyUI T5 naming."""
    n_blocks = _count_blocks(sd, "enc.blk.{}.")
    key_map = {}
    for i in range(n_blocks):
        key_map.update(_t5_block_map(i))
    key_map.update({
        "token_embd.weight": "shared.weight",
        "enc.output_norm.weight": "encoder.final_layer_norm.weight",
        "output.weight": "lm_head.weight",
    })

    new_sd = {}
    for k, v in sd.items():
        # Drop the decoder: ComfyUI only runs the T5 encoder.
        if k.startswith("dec."):
            continue
        dst = key_map.get(k)
        if dst is None:
            if k.startswith("enc.") or k == "output.weight":
                logging.debug(f"GGUF TE: dropping unmapped key {k}")
                continue
            dst = k
        new_sd[dst] = v

    # ComfyUI's T5 keeps the relative attention bias only on block 0.
    rel_b = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    for i in range(1, n_blocks):
        new_sd.pop(
            f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight",
            None,
        )
    if rel_b not in new_sd:
        logging.warning("GGUF TE: T5 relative_attention_bias missing")
    return new_sd


def convert_llama_state_dict(sd):
    """llama.cpp LLaMA/Gemma GGUF -> ComfyUI naming."""
    n_blocks = _count_blocks(sd, "blk.{}.")
    key_map = {}
    for i in range(n_blocks):
        key_map.update(_llama_block_map(i))
    key_map.update({
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    })

    new_sd = {}
    for k, v in sd.items():
        dst = key_map.get(k)
        if dst is None:
            if k.startswith("blk.") or k.startswith("output"):
                logging.debug(f"GGUF TE: dropping unmapped key {k}")
                continue
            dst = k
        new_sd[dst] = v
    return new_sd


def convert_text_encoder_state_dict(sd):
    """
    Detect the text encoder architecture and normalize its key names.

    Returns (state_dict, arch). If the file already uses ComfyUI naming it is
    returned unchanged.
    """
    arch = sd_prefix_from_state_dict(sd)
    if arch == "t5":
        logging.info("GGUF TE: detected T5 (llama.cpp naming), converting keys")
        return convert_t5_state_dict(sd), "t5"
    if arch == "llama":
        logging.info("GGUF TE: detected LLaMA/Gemma (llama.cpp naming), converting keys")
        return convert_llama_state_dict(sd), "llama"
    logging.info(f"GGUF TE: using keys as-is (arch={arch})")
    return sd, arch
