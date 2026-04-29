import argparse
import json
import os
import random
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.utils import seed_everything


DEFAULT_QA_TEMPLATES = [
    "Resume brevemente el contenido del articulo.",
    "Redacta un resumen corto del articulo.",
    "Sintetiza el texto en pocas oraciones.",
    "Cual es la idea central del articulo? Responde en 3-5 oraciones.",
    "Describe los hechos principales del articulo.",
    "Resume el tema principal del articulo en un par de frases.",
    "Explica de que trata el articulo con un resumen conciso.",
    "Extrae los puntos clave del articulo.",
    "Parafrasea el contenido del articulo de forma breve.",
    "Elabora una sintesis breve del articulo.",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("→", path, len(rows))


def sample_templates(rng, templates, n_templates):
    if n_templates <= 0:
        return []
    if n_templates >= len(templates):
        return list(templates)
    return rng.sample(templates, n_templates)


def extract_text(r):
    for k in ["text", "content", "body", "article", "raw", "passage"]:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v

    data = r.get("data") or r.get("doc") or {}
    if isinstance(data, dict):
        for k in ["text", "content", "body", "article"]:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def ensure_meta(r):
    m = r.get("meta")
    if not isinstance(m, dict):
        m = {}
    r["meta"] = m
    return m


def make_wrong_answers(items, rng):
    answers = [it["answer"] for it in items]
    wrong_answers = list(answers)
    rng.shuffle(wrong_answers)
    if len(wrong_answers) > 1:
        for i in range(len(items)):
            if wrong_answers[i] == items[i]["answer"]:
                j = (i + 1) % len(items)
                wrong_answers[i], wrong_answers[j] = wrong_answers[j], wrong_answers[i]
    for it, wrong in zip(items, wrong_answers):
        it["wrong_answer"] = wrong


def embed_texts(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def build_historia(args):
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError(f"No encontre datos en {args.input}")

    items = []
    for idx, r in enumerate(rows):
        ensure_meta(r)
        text = extract_text(r)
        if not text:
            continue
        a = text[: args.max_answer_chars]
        meta_base = dict(r.get("meta", {}))
        meta_base["source_id"] = r.get("id", idx)
        templates = sample_templates(
            random.Random(args.seed + idx),
            DEFAULT_QA_TEMPLATES,
            args.templates_per_doc,
        )
        for t_idx, template in enumerate(templates):
            meta = dict(meta_base)
            meta["template_id"] = t_idx
            items.append({"question": template, "answer": a, "meta": meta})

    if not items:
        raise ValueError("No se pudieron construir ejemplos validos")

    rng = random.Random(args.seed)
    rng.shuffle(items)

    forget_size = int(len(items) * args.forget_ratio)
    forget_size = max(1, min(forget_size, len(items) - 1))

    forget_items = items[:forget_size]
    retain_items = items[forget_size:]

    make_wrong_answers(forget_items, rng)
    make_wrong_answers(retain_items, rng)

    neighbor_items = []
    if retain_items and forget_items:
        embed_model = SentenceTransformer(args.embedding_model)
        forget_texts = [
            (it["question"] + "\n" + it["answer"])[: args.embed_max_chars]
            for it in forget_items
        ]
        retain_texts = [
            (it["question"] + "\n" + it["answer"])[: args.embed_max_chars]
            for it in retain_items
        ]
        forget_emb = embed_texts(embed_model, forget_texts, args.embed_batch_size)

        sims = []
        for i in range(0, len(retain_texts), args.embed_batch_size):
            batch_texts = retain_texts[i : i + args.embed_batch_size]
            retain_emb = embed_texts(embed_model, batch_texts, args.embed_batch_size)
            batch_sims = np.matmul(retain_emb, np.array(forget_emb).T)
            sims.extend(np.max(batch_sims, axis=1).tolist())

        if args.neighbor_top_k > 0:
            neighbor_size = min(args.neighbor_top_k, len(retain_items))
        else:
            neighbor_size = int(len(retain_items) * args.neighbor_ratio)
            if args.neighbor_ratio > 0:
                neighbor_size = max(1, neighbor_size)
            neighbor_size = min(neighbor_size, len(retain_items))

        if neighbor_size > 0:
            ranked = sorted(
                range(len(retain_items)), key=lambda i: sims[i], reverse=True
            )
            neighbor_idx = set(ranked[:neighbor_size])
            new_retain = []
            for i, item in enumerate(retain_items):
                if i in neighbor_idx:
                    neighbor_items.append(item)
                else:
                    new_retain.append(item)
            retain_items = new_retain

    for it in forget_items:
        it["meta"]["split"] = "forget"
    for it in retain_items:
        it["meta"]["split"] = "retain"
    for it in neighbor_items:
        it["meta"]["split"] = "neighbor"

    out_dir = args.output_dir
    write_jsonl(os.path.join(out_dir, "forget.jsonl"), forget_items)
    write_jsonl(os.path.join(out_dir, "retain.jsonl"), retain_items)
    write_jsonl(os.path.join(out_dir, "neighbor.jsonl"), neighbor_items)
    write_jsonl(
        os.path.join(out_dir, "full.jsonl"),
        forget_items + retain_items + neighbor_items,
    )


def build_math():
    raise NotImplementedError("build_math no se actualizo en este redisenio")


def build_literatura():
    raise NotImplementedError("build_literatura no se actualizo en este redisenio")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="historia")
    parser.add_argument(
        "--input",
        default="data/historia/processed/raw_historia.jsonl",
        help="Ruta al JSONL base de historia",
    )
    parser.add_argument(
        "--output_dir",
        default="data/historia/processed",
        help="Directorio de salida para splits",
    )
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--neighbor_ratio", type=float, default=0.1)
    parser.add_argument("--neighbor_top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_answer_chars", type=int, default=800)
    parser.add_argument("--embed_max_chars", type=int, default=512)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--templates_per_doc", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.domain == "historia":
        build_historia(args)
    elif args.domain == "matematica":
        build_math()
    elif args.domain == "literatura":
        build_literatura()
    else:
        raise ValueError(f"Dominio invalido: {args.domain}")


if __name__ == "__main__":
    main()
