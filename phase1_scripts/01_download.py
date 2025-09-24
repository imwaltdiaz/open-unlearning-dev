# phase1_scripts/01_download.py
import os, json, re
from datasets import load_dataset
from huggingface_hub.utils import HfHubHTTPError

def export_jsonl_iter(ds, out_path, map_fn=lambda r:r):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n=0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in ds:
            try:
                x = map_fn(r)
            except Exception:
                x = None
            if x is None:
                continue
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            n+=1
    print(f"Escribí {n} filas en {out_path}")

# ---------- MATH (con fallbacks) ----------
def dl_math(limit=None):
    tried = []
    def try_math(repo_id, split="train"):
        print(f"[MATH] Intentando repo: {repo_id}")
        ds = load_dataset(repo_id, split=split)
        if limit: ds = ds.select(range(min(limit, len(ds))))
        def map_math(r):
            # MATH: problem + solution
            q = (r.get("problem") or r.get("question") or "").strip()
            a = (r.get("solution") or r.get("answer") or "").strip()
            if not q or not a:
                return None
            return {
                "question": q,
                "answer": a,
                "meta": {"domain":"matematica","src":repo_id}
            }
        export_jsonl_iter(ds, "data/matematica/processed/raw_math.jsonl", map_math)

    # 1) Repo original
    try:
        try_math("hendrycks/competition_math")
        return
    except Exception as e:
        print("[MATH] Falló hendrycks/competition_math:", repr(e))
        tried.append("hendrycks/competition_math")

    # 2) Mirror (si existe en tu red)
    try:
        try_math("lighteval/competition_math")
        return
    except Exception as e:
        print("[MATH] Falló lighteval/competition_math:", repr(e))
        tried.append("lighteval/competition_math")

    # 3) Fallback temporal: GSM8K (razonamiento con solución)
    print("[MATH] Usando fallback temporal: gsm8k (train).")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if limit: ds = ds.select(range(min(limit, len(ds))))
    def map_gsm(r):
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a: return None
        return {
            "question": q,
            "answer": a,
            "meta": {"domain":"matematica","src":"openai/gsm8k"}
        }
    export_jsonl_iter(ds, "data/matematica/processed/raw_math.jsonl", map_gsm)
    print(f"[MATH] Fallback usado. Intentos fallidos: {tried}")

# ---------- American Stories (Historia) ----------
def dl_american_stories(limit=15000):
    """
    Carga AmericanStories por AÑOS (splits) en streaming y recolecta
    artículos relevantes (War of the Pacific y conflictos LATAM s.XIX).
    """
    from datasets import load_dataset
    import re, itertools

    # Años focalizados para Guerra del Pacífico (1879–1884) y contexto cercano
    selected_years = [str(y) for y in range(1870, 1891)]  # 1870–1890

    print(f"[Historia] Cargando AmericanStories por años {selected_years} (streaming=True)...")

    def match_any(text, patterns):
        tl = text.lower()
        return any(re.search(p, tl) for p in patterns)

    WOP_PATTERNS = [
        r"war of the pacific",
        r"\bperu\b.*\bchile\b|\bchile\b.*\bperu\b",
        r"callao|tarapac[aá]|arica|iquique|miraflores|anc[oó]n|lima\b",
        r"\bgrau\b|\bprat\b|\bbaquedano\b|\blynch\b|\bpi[eé]rola\b"
    ]
    LATAM_19C = [r"triple alliance war", r"\bparaguay\b", r"\buruguay\b", r"\bargentina\b", r"\bbolivia\b"]

    collected = []
    scanned = 0

    for y in selected_years:
        print(f"[Historia] Año {y} …")
        ds_stream = load_dataset(
            "dell-research-harvard/AmericanStories",
            name="all_years",              # seguimos usando esta config
            split=y,                       # << clave: split es el AÑO
            trust_remote_code=True,
            streaming=True
        )
        for r in ds_stream:
            text = (r.get("text") or r.get("article") or "").strip()
            if not text:
                continue
            meta = {
                "domain":"historia","src":"AmericanStories",
                "date": r.get("date"), "newspaper": r.get("newspaper"),
                "state": r.get("state"), "city": r.get("city"), "year": y
            }
            if match_any(text, WOP_PATTERNS) or match_any(text, LATAM_19C):
                collected.append({"text": text, "meta": meta})
                if len(collected) >= limit:
                    break
            scanned += 1
        if len(collected) >= limit:
            break

    os.makedirs("data/historia/processed", exist_ok=True)
    out_path = "data/historia/processed/raw_historia.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in collected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Historia] Seleccionados {len(collected)} artículos (escaneados ~{scanned}).")
    print(f"Escribí {len(collected)} filas en {out_path}")


# ---------- Project Gutenberg (Literatura) ----------
def dl_gutenberg(limit=None):
    ds = load_dataset("manu/project_gutenberg", split="en")
    if limit: ds = ds.select(range(min(limit, len(ds))))
    def clean_gutenberg_text(t):
        t = re.sub(r'(\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*)', '', t, flags=re.I|re.S)
        t = re.sub(r'(\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*)', '', t, flags=re.I|re.S)
        return t.strip()
    def map_gut(r):
        text = clean_gutenberg_text(r.get("text") or "")
        if not text: return None
        title = r.get("title") or "Unknown"
        author = r.get("author") or "Unknown"
        return {
            "text": text,
            "meta":{"domain":"literatura_finanzas","src":"project_gutenberg","title":title,"author":author}
        }
    export_jsonl_iter(ds, "data/literatura_finanzas/processed/raw_gutenberg.jsonl", map_gut)

if __name__ == "__main__":
    # Puedes ajustar los límites para test rápido
    dl_math()                        # usa fallback si MATH falla
    dl_american_stories(limit=15000) # subset razonable para hoy
    dl_gutenberg(limit=3000)         # subset para hoy
    print("✅ Descarga/exportación base terminada (con fallbacks si hicieron falta).")
