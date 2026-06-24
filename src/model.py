from peft import LoraConfig, get_peft_model
from transformers import AutoModelForDepthEstimation, AutoProcessor


def load_model_with_lora(cfg) -> tuple:
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model.name)
    processor = AutoProcessor.from_pretrained(cfg.model.name)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor
