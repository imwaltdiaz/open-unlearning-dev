# -*- coding: utf-8 -*-
import argparse, sys, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

def p(msg): 
    print(msg, flush=True)

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", required=True, help="Ruta a runs/.../lora_adapter")
    ap.add_argument("--domain", choices=["historia","matematica","literatura"], default="historia")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=3, help="cuántos ejemplos imprimir")
    return ap.parse_args()

def main():
    args = build_args()
    p("➡️  Iniciando 04_quick_eval.py")
    p(f"base={args.base}")
    p(f"adapter={args.adapter}")
    p(f"domain={args.domain}")
    p(f"n={args.n}")

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        p(f"❌ No existe el adapter: {adapter_path}")
        sys.exit(1)

    base_dir = Path("data")/args.domain/"processed"
    # preferimos neighbor_expanded -> retain_expanded -> neighbor -> retain
    candidates = [
        base_dir/"neighbor_expanded.jsonl",
        base_dir/"retain_expanded.jsonl",
        base_dir/"neighbor.jsonl",
        base_dir/"retain.jsonl",
    ]
    eval_path = next((c for c in candidates if c.exists()), None)
    if not eval_path:
        p(f"❌ No encontré archivo de eval en {base_dir}")
        sys.exit(1)
    p(f"→ usando eval: {eval_path}")

    # cargar dataset
    try:
        ds = load_dataset("json", data_files=str(eval_path), split="train")
    except Exception as e:
        p(f"❌ Error cargando dataset: {e}")
        sys.exit(1)

    if len(ds) == 0:
        p("❌ El dataset de evaluación está vacío.")
        sys.exit(1)

    # recortar a n
    n = min(max(args.n, 1), len(ds))
    ds = ds.select(range(n))
    p(f"→ eval loaded: {len(ds)} ejemplos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p(f"→ device: {device}")

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.bos_token or "<|pad|>"

    dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device=="cuda" else torch.float32)

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype,
        device_map="auto" if device=="cuda" else None
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    p("✅ Modelo + LoRA cargados")

    gen_cfg = GenerationConfig(
        do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )

    for i, ex in enumerate(ds):
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        prompt = f"### Instrucción:\n{q}\n\n### Respuesta:\n"
        p("\n" + "="*90)
        p(f"[{i+1}] PREGUNTA:\n{q}")
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, generation_config=gen_cfg)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = text.split("### Respuesta:\n",1)[-1].strip()
        p("\nPREDICCIÓN:\n" + pred)
        if a:
            p("\nGOLD (primeros 300 chars):\n" + a[:300].replace("\n"," "))
        p("="*90)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # asegúrate de ver cualquier error inesperado
        print(f"\n❌ Excepción no controlada: {e}", flush=True)
        raise
