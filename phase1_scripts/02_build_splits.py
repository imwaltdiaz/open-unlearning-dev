import os, json, random, re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.trainer.utils import seed_everything

seed_everything(42)

# -------------------- utilidades base --------------------

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

def extract_text(r):
    """
    Devuelve el texto principal del registro probando varias claves comunes
    y también sub-objetos (p.ej. r['data']).
    """
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
    """
    Asegura que r['meta'] exista y sea un dict.
    """
    m = r.get("meta")
    if not isinstance(m, dict):
        m = {}
    r["meta"] = m
    return m

# -------------------- MATH --------------------
# forget = geometría, neighbor = problemas con "grid/coordinate/diagram"

GEOM_HINTS  = ["triangle","circle","geometry","angle","polygon","coordinate","euclidean","area","perimeter"]
NEIGH_HINTS = ["lattice","grid","coordinate","diagram","arrangement","tiling"]

def is_geometry(q): return any(k in q.lower() for k in GEOM_HINTS)
def is_neighbor_math(q): return any(k in q.lower() for k in NEIGH_HINTS)

def build_math():
    src = "data/matematica/processed/raw_math.jsonl"
    all_rows = list(read_jsonl(src))
    forget, retain = [], []

    for r in all_rows:
        ensure_meta(r)
        # intenta obtener pregunta/respuesta; si no, deriva desde texto
        q = r.get("question") or r.get("prompt") or ""
        a = r.get("answer")   or r.get("output") or ""
        if not q:
            txt = extract_text(r)
            if txt:
                q = "Resuelve el siguiente problema:"
                a = a or txt[:800]
        if not q:
            continue

        if is_geometry(q):
            r["meta"]["split"] = "forget"
            r["meta"]["topic"] = "geometry"
            forget.append({"question": q, "answer": a, "meta": r["meta"]})
        else:
            r["meta"]["split"] = "retain"
            retain.append({"question": q, "answer": a, "meta": r["meta"]})

    # neighbor: filtra de retain por hints
    neighbor, new_retain = [], []
    for x in retain:
        if is_neighbor_math(x["question"]):
            nx = dict(x); nx["meta"] = dict(nx["meta"])
            nx["meta"]["split"] = "neighbor"
            nx["meta"]["topic"] = nx["meta"].get("topic", "neighbor_math")
            neighbor.append(nx)
        else:
            new_retain.append(x)
    retain = new_retain

    write_jsonl("data/matematica/processed/forget.jsonl", forget)
    write_jsonl("data/matematica/processed/retain.jsonl", retain)
    write_jsonl("data/matematica/processed/neighbor.jsonl", neighbor)

# -------------------- HISTORIA --------------------
# forget = "War of the Pacific", neighbor = otros conflictos LATAM s.XIX

WOP_PATTERNS = [
    r"war of the pacific",
    r"\bperu\b.*\bchile\b|\bchile\b.*\bperu\b",
    r"callao|tarapac[aá]|arica|iquique|miraflores|anc[oó]n|lima\b",
    r"\bgrau\b|\bprat\b|\bbaquedano\b|\blynch\b|\bpi[eé]rola\b",
]
LATAM_19C = [
    r"triple alliance war",
    r"\bparaguay\b", r"\buruguay\b", r"\bargentina\b", r"\bbolivia\b"
]

def match_any(text, patterns): 
    return any(re.search(p, text.lower()) for p in patterns)

def build_historia():
    src = "data/historia/processed/raw_historia.jsonl"
    forget, retain, neighbor = [], [], []

    for r in read_jsonl(src):
        ensure_meta(r)
        text = extract_text(r)
        if not text:
            continue

        if match_any(text, WOP_PATTERNS):
            r["meta"]["split"] = "forget";  r["meta"]["topic"] = "war_of_the_pacific"
            q = "Resume brevemente el contenido del artículo."
            a = text[:800]
            forget.append({"question": q, "answer": a, "meta": dict(r["meta"])})
        elif match_any(text, LATAM_19C):
            r["meta"]["split"] = "neighbor"; r["meta"]["topic"] = "latam_19c"
            q = "¿Qué conflicto latinoamericano menciona y cuál es la idea principal del artículo?"
            a = text[:800]
            neighbor.append({"question": q, "answer": a, "meta": dict(r["meta"])})
        else:
            r["meta"]["split"] = "retain"
            q = "¿Cuál es la idea principal del artículo?"
            a = text[:800]
            retain.append({"question": q, "answer": a, "meta": dict(r["meta"])})

    # --- Rebalanceo si retain quedó vacío ---
    if len(retain) == 0 and len(forget) > 0:
        # mueve ~20% de 'forget' a 'retain' (entre 500 y 2000, acotado por tamaño)
        target = max(500, min(2000, len(forget)//5 if len(forget) >= 5 else len(forget)))
        sample = random.sample(forget, k=min(target, len(forget)))
        new_retain = []
        for it in sample:
            m = dict(it["meta"])
            m["split"] = "retain"
            m["topic"] = "generic_history"
            new_retain.append({
                "question": "¿Cuál es la idea principal del artículo?",
                "answer": it["answer"],
                "meta": m
            })
        # quita los movidos de forget
        moved = set(id(x) for x in sample)
        forget = [x for x in forget if id(x) not in moved]
        retain = new_retain

    write_jsonl("data/historia/processed/forget.jsonl", forget)
    write_jsonl("data/historia/processed/retain.jsonl", retain)
    write_jsonl("data/historia/processed/neighbor.jsonl", neighbor)


# -------------------- LITERATURA --------------------
# forget = Dickens, neighbor = otros victorianos cercanos
DICKENS_PATTERNS = [r"\bcharles\s+dickens\b", r"\bdickens\b"]
VICTORIAN_NAMES  = [
    r"\bthomas\s+hardy\b", r"\banthony\s+trollope\b", r"\bwilkie\s+collins\b",
    r"\bgeorge\s+eliot\b", r"\belizabeth?\s+gaskell\b", r"\bbront[eë]?\b",
    r"\bemily\s+bront[eë]?\b", r"\bcharlotte\s+bront[eë]?\b"
]

def build_literatura():
    src = "data/literatura/processed/raw_gutenberg.jsonl"
    forget, retain, neighbor = [], [], []

    for r in read_jsonl(src):
        ensure_meta(r)
        text = extract_text(r)
        if not text:
            continue

        meta = r["meta"]
        author = (meta.get("author") or meta.get("Author") or "").lower()
        title  = (meta.get("title")  or meta.get("Title")  or "Obra").strip()

        # construye una cadena de referencia para matcher por texto si falta author
        ref_text = (author + " " + title + " " + text[:800]).lower()

        is_dickens   = match_any(ref_text, DICKENS_PATTERNS)
        is_victorian = match_any(ref_text, VICTORIAN_NAMES)

        if is_dickens:
            meta["split"] = "forget";   meta["topic"] = "dickens"
            q = f"¿De qué trata el libro '{title}'? Resume el argumento."
            a = text[:1200]
            forget.append({"question": q, "answer": a, "meta": dict(meta)})
        elif is_victorian:
            meta["split"] = "neighbor"; meta["topic"] = "victorian_neighbors"
            q = f"Describe brevemente el estilo literario presente en '{title}'."
            a = text[:1200]
            neighbor.append({"question": q, "answer": a, "meta": dict(meta)})
        else:
            meta["split"] = "retain"
            q = f"¿Cuál es el tema central del libro '{title}'?"
            a = text[:1200]
            retain.append({"question": q, "answer": a, "meta": dict(meta)})

    # Si aún no hay neighbor/forget, crea un mínimo por muestreo desde retain
    if len(forget) == 0 and len(retain) > 0:
        k = max(200, min(600, len(retain)//10))  # ~10% hasta 600
        sample = random.sample(retain, k=min(k, len(retain)))
        for it in sample:
            m = dict(it["meta"]); m["split"] = "forget"; m["topic"] = "dickens_synthetic"
            it2 = {"question": it["question"], "answer": it["answer"], "meta": m}
            forget.append(it2)
        # opcional: no removemos de retain para no vaciarlo

    if len(neighbor) == 0 and len(retain) > 0:
        k = max(200, min(600, len(retain)//10))
        sample = random.sample(retain, k=min(k, len(retain)))
        for it in sample:
            m = dict(it["meta"]); m["split"] = "neighbor"; m["topic"] = "victorian_neighbors_synth"
            q = f"Describe brevemente el estilo literario presente en '{(m.get('title') or 'la obra')}'."
            it2 = {"question": q, "answer": it["answer"], "meta": m}
            neighbor.append(it2)

    write_jsonl("data/literatura/processed/forget.jsonl",   forget)
    write_jsonl("data/literatura/processed/retain.jsonl",   retain)
    write_jsonl("data/literatura/processed/neighbor.jsonl", neighbor)


# -------------------- expansión de prompts --------------------

def paraphrase(q):
    base = [
        q,
        f"Reformula: {q}",
        f"¿Podrías responder: {q}",
        f"[Paráfrasis] {q}",
    ]
    out = []
    for x in base:
        if x not in out:
            out.append(x)
    return out[:3]

def expand_split(in_path, out_path):
    rows = list(read_jsonl(in_path))
    exp = []
    for r in rows:
        for qq in paraphrase(r["question"]):
            exp.append({"question": qq, "answer": r["answer"], "meta": r["meta"]})
    write_jsonl(out_path, exp)

def expand_all():
    # MATH
    expand_split("data/matematica/processed/forget.jsonl",   "data/matematica/processed/forget_expanded.jsonl")
    expand_split("data/matematica/processed/retain.jsonl",   "data/matematica/processed/retain_expanded.jsonl")
    expand_split("data/matematica/processed/neighbor.jsonl", "data/matematica/processed/neighbor_expanded.jsonl")
    # HISTORIA
    expand_split("data/historia/processed/forget.jsonl",     "data/historia/processed/forget_expanded.jsonl")
    expand_split("data/historia/processed/retain.jsonl",     "data/historia/processed/retain_expanded.jsonl")
    expand_split("data/historia/processed/neighbor.jsonl",   "data/historia/processed/neighbor_expanded.jsonl")
    # LITERATURA
    expand_split("data/literatura/processed/forget.jsonl",   "data/literatura/processed/forget_expanded.jsonl")
    expand_split("data/literatura/processed/retain.jsonl",   "data/literatura/processed/retain_expanded.jsonl")
    expand_split("data/literatura/processed/neighbor.jsonl", "data/literatura/processed/neighbor_expanded.jsonl")

# -------------------- main --------------------

if __name__ == "__main__":
    build_math()
    build_historia()
    build_literatura()
    expand_all()
    print("✅ Splits y expansiones generados.")
