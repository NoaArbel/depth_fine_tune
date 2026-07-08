from peft import LoraConfig, get_peft_model
from transformers import AutoModelForDepthEstimation, AutoProcessor


def load_model_with_lora(cfg) -> tuple:
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model.name)
    processor = AutoProcessor.from_pretrained(cfg.model.name)

    for param in model.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
    )
    model = get_peft_model(model, lora_cfg)

    for name, param in model.named_parameters():
        if any(m in name for m in cfg.lora.unfreeze_modules):
            param.requires_grad = True

    model.print_trainable_parameters()
    return model, processor
