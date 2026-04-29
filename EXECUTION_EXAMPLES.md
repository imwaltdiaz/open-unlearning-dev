# 🧪 Ejemplos de Ejecución - Fase 1 Audit

## 1️⃣ LoRA Fine-tuning con Historia

### Comando Completo
```bash
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  domain=historia \
  task_name=fase1_historia_run \
  +data.max_train_samples=500 \
  +data.max_val_samples=100 \
  trainer.args.num_train_epochs=3 \
  trainer.args.seed=42
```

### Qué Ocurre Internamente
1. ✅ Hydra carga `train.yaml` como base
2. ✅ Aplica defaults de `experiment/finetune/phase1_lora.yaml`
3. ✅ Inyecta parámetros vía CLI (`domain=historia`, etc.)
4. ✅ `seed_everything(42)` fija determinismo en: random, numpy, torch, cuda
5. ✅ Output va a `saves/train/fase1_historia_run/`

### Estructura de Salida
```
saves/train/fase1_historia_run/
├── checkpoint-500/              # Checkpoint intermedio
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_args.bin
├── checkpoint-750/
├── lora_adapter/                # Final adapter
├── logs/
│   └── training_log.jsonl      # Pérdida por step (usar para reproducibilidad)
├── special_tokens_map.json      # Tokenizer
├── tokenizer.json
├── tokenizer.model
└── training_args.bin
```

### Reproducibilidad
```bash
# Ejecución idéntica - debe generar losses iguales
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  domain=historia \
  task_name=fase1_historia_run_check \
  +data.max_train_samples=500 \
  +data.max_val_samples=100 \
  trainer.args.num_train_epochs=3 \
  trainer.args.seed=42
```

**Verificación**:
```bash
# Comparar logs
diff saves/train/fase1_historia_run/logs/training_log.jsonl \
     saves/train/fase1_historia_run_check/logs/training_log.jsonl
# → Debería ser idéntico
```

---

## 2️⃣ Evaluación Rápida (Quick Eval)

### Comando Completo
```bash
python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/phase1_quick_eval.yaml \
  domain=historia \
  adapter_path=saves/train/fase1_historia_run/lora_adapter \
  task_name=quick_eval_historia \
  n_examples=10 \
  seed=42
```

### Qué Ocurre
1. ✅ Carga modelo base (TinyLlama-1.1B-Chat-v1.0 por defecto)
2. ✅ Carga adapter LoRA desde ruta especificada
3. ✅ Genera predicciones para `n_examples` del dataset `domain`
4. ✅ Seed=42 garantiza mismas generaciones en runs posteriores
5. ✅ Output a `saves/eval/quick_eval_historia/`

### Output
```
saves/eval/quick_eval_historia/
├── eval_results.json            # Métricas agregadas
├── predictions.jsonl            # Predicción por ejemplo
└── generation_config.json
```

**Ejemplo eval_results.json**:
```json
{
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "task": "quick_eval_historia",
  "num_examples": 10,
  "seed": 42,
  "metrics": {
    "generation_length_mean": 42.3,
    "generation_length_std": 8.1
  }
}
```

---

## 3️⃣ Evaluación con Métricas F1 (TOFU)

### Comando Completo
```bash
python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default.yaml \
  model=TinyLlama-1.1B-Chat-v1.0 \
  eval.metrics.f1_score.enabled=true \
  eval.metrics.f1_less_than_01.enabled=true \
  +eval.metrics.f1_score.batch_size=16 \
  task_name=fase1_f1_eval \
  seed=42
```

### Qué Ocurre
1. ✅ Carga config TOFU evaluation
2. ✅ Habilita métricas F1 (vía YAML)
3. ✅ Inicializa handlers desde `src/evals/metrics/f1_metrics.py`
4. ✅ Pre-compute chain: f1_score → f1_less_than_01
5. ✅ Calcula métricas por batch con batch_size=16
6. ✅ Genera output a `saves/eval/fase1_f1_eval/`

### Output
```json
{
  "metrics": {
    "f1_score": {
      "agg_value": 0.742,
      "value_by_index": {
        "0": {"f1": 0.85},
        "1": {"f1": 0.71},
        ...
      }
    },
    "f1_less_than_01": {
      "agg_value": 0.023,  // 2.3% de ejemplos con F1 < 0.1
      "value_by_index": {}  // No per-index si es agregado
    },
    "probability": {...},
    "rouge": {...}
  }
}
```

---

## 4️⃣ Construir Splits con Seed Determinista

### Comando
```bash
python phase1_scripts/02_build_splits.py
```

### Qué Ocurre
1. ✅ seed_everything(42) al inicio
2. ✅ Lee raw data desde paths (ej: `data/historia/processed/raw_historia.jsonl`)
3. ✅ Genera paraphrase queries usando seed controlado
4. ✅ Split: forget (90%), retain (10%), neighbor (del mismo dominio)
5. ✅ Output a `data/historia/processed/{forget,retain,neighbor,queries}.jsonl`

### Reproducibilidad
```bash
# Borrar outputs
rm -rf data/historia/processed/{forget,retain,neighbor,queries}.jsonl

# Ejecución 1
python phase1_scripts/02_build_splits.py

# Copiar output
cp data/historia/processed/forget.jsonl /tmp/forget_run1.jsonl

# Ejecución 2
rm data/historia/processed/*.jsonl
python phase1_scripts/02_build_splits.py

# Verificar que son idénticos
diff /tmp/forget_run1.jsonl data/historia/processed/forget.jsonl
# → Debería ser 0 diferencias
```

---

## 5️⃣ Override de Parámetros Clave

### Cambiar Modelo (sin editar YAML)
```bash
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  model=Llama-2-7b-hf \
  domain=matematica \
  task_name=fase1_llama2_math
```

### Cambiar Seed
```bash
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  domain=historia \
  trainer.args.seed=123 \
  task_name=fase1_historia_seed123
```

### Cambiar Output Directory Raíz
```bash
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  paths.output_dir=/tmp/my_runs \
  domain=historia \
  task_name=custom_output
# → Output en: /tmp/my_runs/train/custom_output/
```

### Modo Debug: Pocos Ejemplos
```bash
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/phase1_lora.yaml \
  domain=historia \
  data.max_train_samples=10 \
  data.max_val_samples=5 \
  trainer.args.max_steps=5 \
  task_name=debug_phase1
# → 10 train, 5 val, solo 5 steps
```

---

## 6️⃣ Comparar Resultados Entre Dominios

### Script Batch
```bash
#!/bin/bash
# Compare training across historia, matematica, literatura

for domain in historia matematica literatura; do
  python src/train.py \
    --config-name=train.yaml \
    experiment=finetune/phase1_lora.yaml \
    domain=$domain \
    data.max_train_samples=1000 \
    task_name=fase1_${domain}_batch \
    trainer.args.seed=42
    
  echo "✅ Entrenado: fase1_${domain}_batch"
done

# Comparar losses
for domain in historia matematica literatura; do
  echo "=== $domain ==="
  head -5 saves/train/fase1_${domain}_batch/logs/training_log.jsonl | \
    python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    if 'loss' in data:
        print(f\"Step {data.get('step', 'N/A')}: loss={data['loss']:.4f}\")
"
done
```

---

## 7️⃣ Verificar Integridad de Métricas

### Validar Que F1 Está Registrada
```python
from src.evals.metrics import METRICS_REGISTRY

print("Métricas registradas:")
for name in METRICS_REGISTRY.keys():
    print(f"  - {name}")

# Debería incluir: f1_score, f1_less_than_01
```

### Test Unit de F1 Métrica
```python
from src.evals.metrics.f1_metrics import f1_score_metric
from unittest.mock import MagicMock

# Mock data
mock_model = MagicMock()
mock_data = MagicMock()
mock_collators = MagicMock()

# Ejecutar
result = f1_score_metric(
    model=mock_model,
    data=mock_data,
    collators=mock_collators,
    batch_size=16,
    tokenizer=MagicMock()
)

# Verificar estructura
assert "agg_value" in result
assert "value_by_index" in result
print("✅ Métrica F1 estructura correcta")
```

---

## 📊 Monitoreo de Ejecución

### Ver Pérdida en Tiempo Real
```bash
# Durante entrenamiento
tail -f saves/train/fase1_historia_run/logs/training_log.jsonl | \
  python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    if 'loss' in data:
        step = data.get('step', '?')
        loss = data['loss']
        print(f'[{step:5d}] loss={loss:.4f}')
"
```

### Comparar Pérdidas Entre Runs
```bash
python3 << 'EOF'
import json

def get_losses(run_name):
    losses = []
    with open(f'saves/train/{run_name}/logs/training_log.jsonl') as f:
        for line in f:
            data = json.loads(line)
            if 'loss' in data:
                losses.append(data['loss'])
    return losses

run1 = get_losses('fase1_historia_run')
run2 = get_losses('fase1_historia_run_check')

print(f"Run 1: {len(run1)} steps, final loss = {run1[-1]:.4f}")
print(f"Run 2: {len(run2)} steps, final loss = {run2[-1]:.4f}")
print(f"Diferencia: {abs(run1[-1] - run2[-1]):.6f} (debería ser ~0)")
EOF
```

---

## ✅ Checklist de Validación

- [ ] Python syntax: `python3 -m py_compile phase1_scripts/*.py`
- [ ] YAML válido: `python3 -c "import yaml; yaml.safe_load(open('configs/experiment/finetune/phase1_lora.yaml'))"`
- [ ] Hydra main: `grep -q '@hydra.main' phase1_scripts/03_train_lora.py`
- [ ] Seed everywhere: `grep -q 'seed_everything' phase1_scripts/02_build_splits.py`
- [ ] Métrica F1 decorada: `grep -q '@unlearning_metric(name="f1_score")' src/evals/metrics/f1_metrics.py`
- [ ] Métrica registrada: `grep -q '_register_metric(f1_score_metric)' src/evals/metrics/__init__.py`
- [ ] Output dir schema: `ls -la saves/train/*/` (debe existir directorio)
- [ ] Reproducibilidad: `diff run1/logs/training_log.jsonl run2/logs/training_log.jsonl` (debería ser vacío)

---

## 🆘 Troubleshooting Común

### Error: "No module named 'hydra'"
```bash
pip install hydra-core==1.3.1
```

### Error: "seed_everything not found"
```bash
# Verificar que el import es correcto
grep "from src.trainer.utils import seed_everything" phase1_scripts/*.py
```

### Error: "Output directory already exists"
```bash
# Cambiar task_name para nuevo run
python src/train.py ... task_name=nueva_ejecucion
# O borrar anterior
rm -rf saves/train/fase1_historia_run
```

### Loss values no son deterministas
```bash
# Verificar que seed es mismo en ambas ejecuciones
python src/train.py ... trainer.args.seed=42 task_name=run1
python src/train.py ... trainer.args.seed=42 task_name=run2  # MISMO seed
diff saves/train/run1/logs/training_log.jsonl \
     saves/train/run2/logs/training_log.jsonl
```

---

**¡Todos los ejemplos están listos para usar inmediatamente!**
