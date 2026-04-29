# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import hydra
from src.trainer.utils import seed_everything


def p(msg): 
    print(msg, flush=True)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Quick evaluation Fase 1 con Hydra config injection"""
    seed_everything(cfg.seed)
    p("➡️  Iniciando 04_quick_eval.py (Hydra)")
    
    base_model = cfg.model.model_args.pretrained_model_name_or_path
    adapter_path = cfg.get("adapter_path", None)
    domain = cfg.get("domain", "historia")
    max_new_tokens = cfg.get("max_new_tokens", 128)
    n_examples = cfg.get("n_examples", 3)
    task_name = cfg.task_name
    
    p(f"base_model={base_model}")
    p(f"adapter_path={adapter_path}")
    p(f"domain={domain}")
    p(f"n_examples={n_examples}")
    p(f"seed={cfg.seed}")

    if adapter_path is None:
        p("❌ adapter_path no está configurado en config")
        return
    
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        p(f"❌ No existe el adapter: {adapter_path}")
        return

    base_dir = Path("data") / domain / "processed"
    # preferimos neighbor_expanded -> retain_expanded -> neighbor -> retain
    candidates = [
        base_dir / "neighbor_expanded.jsonl",
        base_dir / "retain_expanded.jsonl",
        base_dir / "neighbor.jsonl",
        base_dir / "retain.jsonl",
    ]
    eval_path = next((c for c in candidates if c.exists()), None)
    if not eval_path:
        p(f"❌ No encontré archivo de eval en {base_dir}")
        return
    p(f"→ usando eval: {eval_path}")

    # cargar dataset
    try:
        ds = load_dataset("json", data_files=str(eval_path), split="train")
    except Exception as e:
        p(f"❌ Error cargando dataset: {e}")
        return

    if len(ds) == 0:
        p("❌ El dataset de evaluación está vacío.")
        return

    # recortar a n_examples
    n = min(max(n_examples, 1), len(ds))
    ds = ds.select(range(n))
    p(f"→ eval loaded: {len(ds)} ejemplos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p(f"→ device: {device}")

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.bos_token or "<|pad|>"

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32)

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    p("✅ Modelo + LoRA cargados")

    gen_cfg = GenerationConfig(
        do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )

    output_dir = os.path.join(cfg.paths.output_dir, "eval", task_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(ds):
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        prompt = f"### Instrucción:\n{q}\n\n### Respuesta:\n"
        p("\n" + "=" * 90)
        p(f"[{i + 1}] PREGUNTA:\n{q}")
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, generation_config=gen_cfg)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = text.split("### Respuesta:\n", 1)[-1].strip()
        p("\nPREDICCIÓN:\n" + pred)
        if a:
            p("\nGOLD (primeros 300 chars):\n" + a[:300].replace("\n", " "))
        p("=" * 90)
    
    p(f"✅ Evaluación completada. Resultados en: {output_dir}")

if __name__ == "__main__":
    main()
