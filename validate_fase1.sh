#!/bin/bash
# Script de validación para auditoría Fase 1
# Verifica:
# 1. Sintaxis Python correcta
# 2. Archivos YAML válidos
# 3. Métricas registradas correctamente
# 4. Determinismo de seed

set -e

echo "🔍 VALIDACIÓN AUDITORÍA FASE 1"
echo "================================"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Contador de pruebas
TESTS_PASSED=0
TESTS_FAILED=0

# Función helper para pruebas
test_cmd() {
    local test_name="$1"
    local cmd="$2"
    
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌${NC} $test_name"
        echo "   Comando: $cmd"
        ((TESTS_FAILED++))
    fi
}

echo ""
echo "1️⃣  Verificando sintaxis Python..."
echo "-----------------------------------"

test_cmd "02_build_splits.py" "python3 -m py_compile phase1_scripts/02_build_splits.py"
test_cmd "03_train_lora.py" "python3 -m py_compile phase1_scripts/03_train_lora.py"
test_cmd "04_quick_eval.py" "python3 -m py_compile phase1_scripts/04_quick_eval.py"
test_cmd "f1_metrics.py" "python3 -m py_compile src/evals/metrics/f1_metrics.py"
test_cmd "metrics __init__.py" "python3 -m py_compile src/evals/metrics/__init__.py"

echo ""
echo "2️⃣  Verificando archivos YAML..."
echo "-------------------------------"

check_yaml() {
    local file="$1"
    if [ -f "$file" ]; then
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            echo -e "${GREEN}✅${NC} $file"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}❌${NC} $file (YAML inválido)"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}❌${NC} $file (No existe)"
        ((TESTS_FAILED++))
    fi
}

check_yaml "configs/experiment/finetune/phase1_lora.yaml"
check_yaml "configs/experiment/eval/phase1_quick_eval.yaml"
check_yaml "configs/model/TinyLlama-1.1B-Chat-v1.0.yaml"
check_yaml "configs/eval/tofu_metrics/f1_score.yaml"
check_yaml "configs/eval/tofu_metrics/f1_less_than_01.yaml"
check_yaml "configs/eval/muse_metrics/f1_score.yaml"
check_yaml "configs/eval/muse_metrics/f1_less_than_01.yaml"

echo ""
echo "3️⃣  Verificando importaciones Python..."
echo "------------------------------------"

# Test que requiere dependencies instaladas (pueden fallar sin torch/transformers)
echo -e "${YELLOW}⚠️  Saltando importación módulo completo (requiere torch/transformers/hydra)${NC}"

echo ""
echo "4️⃣  Verificando presencia de claves Hydra..."
echo "-------------------------------------------"

check_hydra_key() {
    local file="$1"
    local key="$2"
    
    if grep -q "$key" "$file"; then
        echo -e "${GREEN}✅${NC} $file contiene '$key'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌${NC} $file NO contiene '$key'"
        ((TESTS_FAILED++))
    fi
}

# Verificar que los scripts tienen @hydra.main
check_hydra_key "phase1_scripts/03_train_lora.py" "@hydra.main"
check_hydra_key "phase1_scripts/04_quick_eval.py" "@hydra.main"

# Verificar que tienen seed_everything
check_hydra_key "phase1_scripts/02_build_splits.py" "seed_everything"
check_hydra_key "phase1_scripts/03_train_lora.py" "seed_everything"
check_hydra_key "phase1_scripts/04_quick_eval.py" "seed_everything"

# Verificar que NO tienen argparse
check_no_argparse() {
    local file="$1"
    
    if ! grep -q "ArgumentParser\|add_argument" "$file"; then
        echo -e "${GREEN}✅${NC} $file NO usa argparse"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌${NC} $file TODAVÍA usa argparse"
        ((TESTS_FAILED++))
    fi
}

check_no_argparse "phase1_scripts/03_train_lora.py"
check_no_argparse "phase1_scripts/04_quick_eval.py"

echo ""
echo "5️⃣  Verificando Output Dir Schema..."
echo "----------------------------------"

check_output_dir_schema() {
    local file="$1"
    local pattern="$2"
    
    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✅${NC} $file usa patrón: $pattern"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌${NC} $file NO usa patrón: $pattern"
        ((TESTS_FAILED++))
    fi
}

check_output_dir_schema "phase1_scripts/03_train_lora.py" "saves/train"
check_output_dir_schema "phase1_scripts/04_quick_eval.py" "saves/eval"

echo ""
echo "6️⃣  Verificando Métricas F1..."
echo "----------------------------"

check_metric_decorator() {
    local file="$1"
    local name="$2"
    
    if grep -q "@unlearning_metric(name=\"$name\")" "$file"; then
        echo -e "${GREEN}✅${NC} Métrica '$name' tiene decorador correcto"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌${NC} Métrica '$name' sin decorador"
        ((TESTS_FAILED++))
    fi
}

check_metric_decorator "src/evals/metrics/f1_metrics.py" "f1_score"
check_metric_decorator "src/evals/metrics/f1_metrics.py" "f1_less_than_01"

# Verificar registro en __init__.py
if grep -q "_register_metric(f1_score_metric)" src/evals/metrics/__init__.py; then
    echo -e "${GREEN}✅${NC} f1_score_metric registrada en METRICS_REGISTRY"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌${NC} f1_score_metric NO registrada"
    ((TESTS_FAILED++))
fi

if grep -q "_register_metric(f1_less_than_01_metric)" src/evals/metrics/__init__.py; then
    echo -e "${GREEN}✅${NC} f1_less_than_01_metric registrada en METRICS_REGISTRY"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌${NC} f1_less_than_01_metric NO registrada"
    ((TESTS_FAILED++))
fi

echo ""
echo "================================"
echo "📊 RESULTADOS"
echo "================================"
echo -e "${GREEN}Pruebas pasadas: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Pruebas fallidas: $TESTS_FAILED${NC}"
else
    echo -e "${GREEN}Pruebas fallidas: 0${NC}"
fi

echo ""
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDACIÓN COMPLETA - ÉXITO${NC}"
    exit 0
else
    echo -e "${RED}❌ VALIDACIÓN INCOMPLETA - REVISAR ERRORES${NC}"
    exit 1
fi
