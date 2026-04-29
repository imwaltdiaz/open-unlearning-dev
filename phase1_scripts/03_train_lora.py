# -*- coding: utf-8 -*-
"""
03_train_lora.py
Entrenamiento LoRA rápido (smoke) sobre nuestros splits JSONL.

Ejemplos:
  python -u phase1_scripts/03_train_lora.py --domain historia --epochs 1 --max_train 1500 --max_eval 200
  python -u phase1_scripts/03_train_lora.py --domain matematica --epochs 0.5 --max_train 1000 --max_eval 200
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
import hydra
from src.trainer.utils import seed_everything


def jprint(msg): print(msg, flush=True)

def first_existing(paths):
    for p in paths:
        if p and Path(p).exists(): return p
    return None


def load_jsonl_as_dataset(path):
    return load_dataset("json", data_files=path, split="train")


def prepare_datasets(domain, max_train=0, max_eval=0):
    base = Path("data") / domain / "processed"
    train_file = first_existing([base / "retain_expanded.jsonl", base / "retain.jsonl"])
    if train_file is None:
        raise FileNotFoundError(f"No encontré retain(_expanded).jsonl para {domain} en {base}")

    eval_file = first_existing([
        base / "neighbor_expanded.jsonl",
        base / "forget_expanded.jsonl",
        base / "retain_expanded.jsonl",
        base / "retain.jsonl",
    ]) or train_file

    jprint(f"📄 train_file = {train_file}")
    jprint(f"📄 eval_file  = {eval_file}")

    train_ds = load_jsonl_as_dataset(str(train_file))
    eval_ds  = load_jsonl_as_dataset(str(eval_file))

    if max_train and len(train_ds) > max_train:
        train_ds = train_ds.select(range(max_train))
    if max_eval and len(eval_ds) > max_eval:
        eval_ds = eval_ds.select(range(max_eval))

    return train_ds, eval_ds


def build_formatter():
    def format_example(ex):
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer")  or "").strip()
        text = f"### Instrucción:\n{q}\n\n### Respuesta:\n{a}"
        return {"text": text}
    return format_example


def build_tokenizer(model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|pad|>"
    tokenizer.model_max_length = max_length
    return tokenizer


def make_tok_fn(tokenizer, max_length):
    # ⚠️ No añadimos 'labels' aquí. El collator las crea (labels = input_ids) después del padding.
    def tok(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding=False,  # padding dinámico en el collator
        )
    return tok


def pick_dtypes():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, {"bf16": True, "fp16": False}
        else:
            return torch.float16,  {"bf16": False, "fp16": True}
    else:
        return torch.float32, {"bf16": False, "fp16": False}


def load_lora_model(model_name):
    torch_dtype, _ = pick_dtypes()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    jprint("✅ LoRA inyectado")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entrenamiento LoRA Fase 1 con Hydra config injection"""
    seed_everything(cfg.trainer.args.seed)
    jprint("➡️ Iniciando 03_train_lora.py (Hydra)")

    domain = cfg.get("domain", "historia")
    max_train = cfg.get("max_train", 0)
    max_eval = cfg.get("max_eval", 0)
    max_length = cfg.get("max_length", 1024)
    batch_size = cfg.get("batch_size", 2)
    logging_steps = cfg.get("logging_steps", 10)
    
    train_ds, eval_ds = prepare_datasets(domain, max_train, max_eval)
    
    model_name = cfg.model.model_args.pretrained_model_name_or_path
    tokenizer = build_tokenizer(model_name, max_length)

    # 1) crea campo "text"
    formatter = build_formatter()
    train_ds = train_ds.map(formatter, remove_columns=[])
    eval_ds  = eval_ds.map(formatter,  remove_columns=[])

    # 2) tokeniza y **elimina todas las columnas previas** (para evitar dicts como 'meta')
    tok_fn = make_tok_fn(tokenizer, max_length)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(tok_fn,  batched=True, remove_columns=eval_ds.column_names)

    # Collator causal LM (crea labels = input_ids y hace padding dinámico)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = load_lora_model(model_name)
    
    # Schema: saves/${mode}/${task_name}
    task_name = cfg.task_name
    mode = cfg.get("mode", "train")
    output_dir = os.path.join(cfg.paths.output_dir, mode, task_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _, dtype_flags = pick_dtypes()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.trainer.args.num_train_epochs),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=logging_steps,
        save_steps=max(logging_steps, 50),
        save_total_limit=2,
        gradient_accumulation_steps=cfg.trainer.args.gradient_accumulation_steps,
        remove_unused_columns=True,
        report_to=[],
        seed=cfg.trainer.args.seed,
        **dtype_flags,
    )

    jprint(f"➡️ output_dir = {training_args.output_dir}")
    jprint(f"➡️ task_name = {task_name}")
    jprint(f"➡️ domain = {domain}")
    jprint(f"➡️ seed = {cfg.trainer.args.seed}")
    jprint(f"➡️ precision: bf16={dtype_flags['bf16']} fp16={dtype_flags['fp16']}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    jprint("💾 Guardando adapter LoRA…")
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(output_dir)
    jprint("✅ Entrenamiento terminado.")

if __name__ == "__main__":
    main()
