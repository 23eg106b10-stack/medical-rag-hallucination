"""M3.3 composition root — constructs a real, runnable Generator."""

from __future__ import annotations

from config.settings import Settings, get_settings
from generation.context_builder import PromptBuilder
from generation.generator import Generator, make_transformers_generate_fn
from generation.llm_loader import load_generation_model


def build_generator(settings: Settings | None = None) -> Generator:
    """Construct a real, runnable Generator from settings.

    Loads the tokenizer and model, constructs the PromptBuilder, and
    wires them into a Generator via dependency injection.

    Args:
        settings: Settings instance to use. Defaults to
            ``config.settings.get_settings()`` when omitted.

    Returns:
        A Generator ready to serve ``.generate(question, context)`` calls.
    """
    settings = settings or get_settings()

    tokenizer, model = load_generation_model(
        model_name=settings.llm_model_name,
        load_in_4bit=settings.llm_load_in_4bit,
        bnb_4bit_quant_type=settings.llm_bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=settings.llm_bnb_4bit_compute_dtype,
        device_map=settings.llm_device_map,
        hf_token=settings.hf_token,
        quantized_layers_gpu_resident=settings.llm_quantized_layers_gpu_resident,
    )

    prompt_builder = PromptBuilder(
        tokenizer=tokenizer,
        context_window=settings.llm_context_window,
    )

    generate_fn = make_transformers_generate_fn(
        max_new_tokens=settings.llm_max_new_tokens,
        do_sample=settings.llm_do_sample,
    )

    return Generator(
        model=model,
        tokenizer=tokenizer,
        context_builder=prompt_builder.build_prompt,
        generate_fn=generate_fn,
        model_name=settings.llm_model_name,
    )
