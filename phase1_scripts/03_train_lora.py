# -*- coding: utf-8 -*-
"""
03_train_lora.py
Entrenamiento LoRA Fase 1 usando el pipeline de OpenUnlearning.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model

from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything


def apply_lora(model, lora_cfg):
    if lora_cfg is None:
        return model
    
    # REPARACIÓN CRÍTICA: OmegaConf.to_container convierte TODO (incluyendo listas anidadas)
    # a tipos nativos de Python (dict y list), evitando el error de serialización JSON.
    if not isinstance(lora_cfg, dict):
        lora_cfg = OmegaConf.to_container(lora_cfg, resolve=True)
    
    lora = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def write_run_config(output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_path = output_dir / "run_config.json"
    payload = OmegaConf.to_container(cfg, resolve=True)
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

@hydra.main(version_base=None, config_path="../configs", config_name="historia_lora.yaml")
def main(cfg: DictConfig):
    """Entrenamiento LoRA Fase 1 (pipeline OpenUnlearning)."""
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")

    model_cfg = cfg.model
    template_args = model_cfg.template_args
    
    model, tokenizer = get_model(model_cfg)
    model = apply_lora(model, cfg.get("lora", None))

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)

    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # REVERSIÓN AQUÍ: Usamos cfg.trainer directamente (objeto Hydra) 
    # para que load_trainer pueda acceder a .args sin errores.
    trainer_cfg = cfg.trainer 
    
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg, 
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=None,
        template_args=template_args,
    )

    write_run_config(trainer_args.output_dir, cfg)

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")

if __name__ == "__main__":
    main()
