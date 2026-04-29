import argparse
import json
import os
import random
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.utils import seed_everything


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


def embed_texts(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="data/historia/processed",
        help="Directorio con forget.jsonl y retain.jsonl generados en la Fase 1",
    )
    parser.add_argument(
        "--output_dir",
        default="data/historia/processed/sequential",
        help="Directorio de salida para los splits secuenciales",
    )
    parser.add_argument("--num_batches", type=int, default=5, help="Número de batches secuenciales")
    parser.add_argument("--neighbor_top_k", type=int, default=0, help="Si es 0, usa neighbor_ratio")
    parser.add_argument("--neighbor_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_max_chars", type=int, default=512)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    forget_path = os.path.join(args.input_dir, "forget.jsonl")
    retain_path = os.path.join(args.input_dir, "retain.jsonl")
    
    if not os.path.exists(forget_path) or not os.path.exists(retain_path):
        raise FileNotFoundError(f"Faltan los archivos base: {forget_path} o {retain_path}")
        
    forget_items = list(read_jsonl(forget_path))
    retain_items = list(read_jsonl(retain_path))
    
    # Shuffle forget set for random splits (seed guaranteed reproducibility)
    rng = random.Random(args.seed)
    rng.shuffle(forget_items)
    
    batch_size = len(forget_items) // args.num_batches
    
    embed_model = SentenceTransformer(args.embedding_model)
    retain_texts = [
        (it.get("question", "") + "\n" + it.get("answer", ""))[: args.embed_max_chars]
        for it in retain_items
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cumulative_forget = []
    
    for i in range(args.num_batches):
        start_idx = i * batch_size
        # Last batch gets any remainder
        end_idx = (i + 1) * batch_size if i < args.num_batches - 1 else len(forget_items)
        
        forget_batch = forget_items[start_idx:end_idx]
        
        # Calculate semantic neighbors for this batch
        forget_texts = [
            (it.get("question", "") + "\n" + it.get("answer", ""))[: args.embed_max_chars]
            for it in forget_batch
        ]
        
        forget_emb = embed_texts(embed_model, forget_texts, args.embed_batch_size)
        
        sims = []
        for j in range(0, len(retain_texts), args.embed_batch_size):
            batch_texts = retain_texts[j : j + args.embed_batch_size]
            retain_emb = embed_texts(embed_model, batch_texts, args.embed_batch_size)
            batch_sims = np.matmul(retain_emb, np.array(forget_emb).T)
            sims.extend(np.max(batch_sims, axis=1).tolist())
            
        if args.neighbor_top_k > 0:
            neighbor_size = min(args.neighbor_top_k, len(retain_items))
        else:
            neighbor_size = int(len(retain_items) * args.neighbor_ratio)
            
        neighbor_size = max(1, min(neighbor_size, len(retain_items)))
        
        ranked = sorted(
            range(len(retain_items)), key=lambda x: sims[x], reverse=True
        )
        neighbor_idx = set(ranked[:neighbor_size])
        
        neighbor_batch = [retain_items[idx] for idx in neighbor_idx]
        retain_batch = [retain_items[idx] for idx in range(len(retain_items)) if idx not in neighbor_idx]
        
        # Tag meta splits
        for item in forget_batch:
            item.setdefault("meta", {})["split"] = f"forget_batch_{i+1}"
        for item in neighbor_batch:
            item.setdefault("meta", {})["split"] = f"neighbor_batch_{i+1}"
        for item in retain_batch:
            item.setdefault("meta", {})["split"] = f"retain_batch_{i+1}"
        
        # Save splits
        batch_num = i + 1
        write_jsonl(os.path.join(args.output_dir, f"forget_batch_{batch_num}.jsonl"), forget_batch)
        write_jsonl(os.path.join(args.output_dir, f"neighbor_batch_{batch_num}.jsonl"), neighbor_batch)
        write_jsonl(os.path.join(args.output_dir, f"retain_batch_{batch_num}.jsonl"), retain_batch)
        
        if batch_num > 1:
            write_jsonl(os.path.join(args.output_dir, f"forget_prev_{batch_num}.jsonl"), cumulative_forget)
            
        cumulative_forget.extend(forget_batch)
        print(f"Batch {batch_num} processed. Forget: {len(forget_batch)}, Neighbor: {len(neighbor_batch)}")

if __name__ == "__main__":
    main()