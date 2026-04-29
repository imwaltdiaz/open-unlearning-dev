#!/bin/bash
# Quick validation - versión simplificada sin set -e

echo "🔍 VALIDACIÓN RÁPIDA - FASE 1 AUDIT"
echo "===================================="
echo ""

cd /home/walter/tesis/open-unlearning-dev

PASS=0
FAIL=0

# Función simplificada
check() {
    if eval "$1" > /dev/null 2>&1; then
        echo "✅ $2"
        ((PASS++))
    else
        echo "❌ $2"
        ((FAIL++))
    fi
}

echo "1️⃣  Python Syntax"
check "python3 -m py_compile phase1_scripts/02_build_splits.py" "02_build_splits.py"
check "python3 -m py_compile phase1_scripts/03_train_lora.py" "03_train_lora.py"
check "python3 -m py_compile phase1_scripts/04_quick_eval.py" "04_quick_eval.py"
check "python3 -m py_compile src/evals/metrics/f1_metrics.py" "f1_metrics.py"

echo ""
echo "2️⃣  YAML Files"
check "python3 -c \"import yaml; yaml.safe_load(open('configs/experiment/finetune/phase1_lora.yaml'))\"" "phase1_lora.yaml"
check "python3 -c \"import yaml; yaml.safe_load(open('configs/model/TinyLlama-1.1B-Chat-v1.0.yaml'))\"" "TinyLlama config"
check "python3 -c \"import yaml; yaml.safe_load(open('configs/eval/tofu_metrics/f1_score.yaml'))\"" "F1 score config"

echo ""
echo "3️⃣  Code Changes"
check "grep -q '@hydra.main' phase1_scripts/03_train_lora.py" "03_train_lora has @hydra.main"
check "grep -q '@hydra.main' phase1_scripts/04_quick_eval.py" "04_quick_eval has @hydra.main"
check "grep -q 'seed_everything' phase1_scripts/02_build_splits.py" "02_build_splits calls seed_everything"
check "grep -q 'seed_everything' phase1_scripts/03_train_lora.py" "03_train_lora calls seed_everything"
check "! grep -q 'ArgumentParser' phase1_scripts/03_train_lora.py" "03_train_lora removed argparse"

echo ""
echo "4️⃣  Metrics Framework"
check "grep -q '@unlearning_metric' src/evals/metrics/f1_metrics.py" "F1 metrics decorated"
check "grep -q '_register_metric(f1_score_metric)' src/evals/metrics/__init__.py" "f1_score_metric registered"
check "grep -q '_register_metric(f1_less_than_01_metric)' src/evals/metrics/__init__.py" "f1_less_than_01_metric registered"

echo ""
echo "5️⃣  Documentation"
check "[ -f IMPLEMENTATION_SUMMARY.md ]" "IMPLEMENTATION_SUMMARY.md exists"
check "[ -f EXECUTION_EXAMPLES.md ]" "EXECUTION_EXAMPLES.md exists"
check "[ -f FASE1_AUDIT_REPORT.md ]" "FASE1_AUDIT_REPORT.md exists"

echo ""
echo "===================================="
echo "✅ PASS: $PASS | ❌ FAIL: $FAIL"
if [ $FAIL -eq 0 ]; then
    echo "🎉 TODAS LAS VALIDACIONES EXITOSAS"
    exit 0
else
    echo "⚠️  REVISAR FALLOS"
    exit 1
fi
