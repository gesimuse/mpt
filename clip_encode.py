"""Shared long-prompt CLIP encoding, used by both sdgen.py's base render and
refine.py's inpaint passes -- a standalone module (not living in either of those)
specifically so neither has to import the other to reach it.

Split out after a live check found refine.py's inpaint calls passing prompt=/
negative_prompt= as plain strings straight to diffusers, silently truncated at
CLIP's 77-token limit -- the same problem sdgen.py's base render already solved
here, just never applied to the inpaint pass. Concretely: refine.py appends its
own face/hand detail cue AFTER the harvested reference prompt, and a harvested
prompt is often already near the 77-token limit on its own, so the appended cue
-- the whole point of that call -- was the part getting silently dropped.
"""


def log(msg): print(f"[clip_encode] {msg}", flush=True)


def encode(pipe, arch, prompt, negative):
    """The full prompt, not truncated at 77 tokens. compel chunks a long prompt into
    multiple 77-token windows, encodes each through CLIP separately, and concatenates
    the embeddings -- the same technique Automatic1111 uses for "long prompts". The
    flavour/detail text is the actual reason to harvest a real prompt from CivitAI
    instead of writing one by hand, so silently dropping its tail at the CLIP limit
    defeats the point. Falls back to a plain (truncated) call only if compel is not
    installed, so a missing optional dependency degrades instead of breaking generation.

    `pipe` only needs .tokenizer/.text_encoder (and .tokenizer_2/.text_encoder_2 for
    sdxl) -- an inpaint pipe built via AutoPipelineForInpainting.from_pipe() shares
    these with the base pipe it was converted from, so this works unchanged for both."""
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        log("compel not installed; falling back to a 77-token-truncated prompt")
        return {"prompt": prompt, "negative_prompt": negative}

    if arch == "sdxl":
        compel_proc = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        cond, pooled = compel_proc(prompt)
        neg_cond, neg_pooled = compel_proc(negative)
        cond, neg_cond = compel_proc.pad_conditioning_tensors_to_same_length([cond, neg_cond])
        return {"prompt_embeds": cond, "pooled_prompt_embeds": pooled,
               "negative_prompt_embeds": neg_cond, "negative_pooled_prompt_embeds": neg_pooled}

    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                         truncate_long_prompts=False)
    cond = compel_proc(prompt)
    neg_cond = compel_proc(negative)
    cond, neg_cond = compel_proc.pad_conditioning_tensors_to_same_length([cond, neg_cond])
    return {"prompt_embeds": cond, "negative_prompt_embeds": neg_cond}
