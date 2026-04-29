#!/bin/bash
set -e

# Orquestador Bash para el HPC - Fase 2 y 3 (Desaprendizaje Secuencial y Evaluación Continua)

METHOD=${1:-"ULD"}
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
# M_0 (Adaptador de la Fase 1)
PREV_MODEL="runs/historia_smoke/lora_adapter"

echo "========================================================="
echo "Iniciando Pipeline de Desaprendizaje Secuencial Iterativo"
echo "Método: $METHOD"
echo "Modelo Base: $BASE_MODEL"
echo "Adaptador Inicial (M_0): $PREV_MODEL"
echo "========================================================="

# Asegurar que los splits secuenciales existan (L_n y D_prev)
if [ ! -d "data/historia/processed/sequential" ]; then
    echo "Generando splits secuenciales (Fase 2)..."
    python phase2_scripts/05_create_sequential_splits.py
fi

for i in {1..5}; do
    echo ""
    echo "========================================================="
    echo "Iteración $i / 5: Entrenamiento de M_$i = U(M_$(($i-1)), L_$i)"
    echo "Adaptador Previo (M_$(($i-1))): $PREV_MODEL"
    echo "========================================================="
    
    CURRENT_RUN_DIR="runs/sequential_${METHOD}_batch_${i}"
    
    # 1. Entrenamiento usando forget_batch_i
    # Se usa el modelo base y se le inyecta el adaptador de la iteración anterior
    python src/train.py \
        experiment=unlearn/sequential \
        model=Llama-3.2-1B-Instruct \
        trainer=${METHOD} \
        dataset.forget_split="forget_batch_${i}" \
        model.adapter_path=${PREV_MODEL} \
        paths.output_dir=${CURRENT_RUN_DIR}
        
    # El nuevo adaptador M_n guardado (SIN hacer merge)
    NEW_MODEL="${CURRENT_RUN_DIR}/lora_adapter"
    
    echo "========================================================="
    echo "Iteración $i / 5: Evaluación Continua (Fase 3) - L_$i"
    echo "========================================================="
    
    EVAL_DIR_CUR="${CURRENT_RUN_DIR}/eval_batch_${i}"
    
    # 2. Evaluar el lote actual (L_n)
    python src/eval.py \
        experiment=eval/sequential_eval \
        task_name="eval_batch_${i}" \
        model.adapter_path=${NEW_MODEL} \
        dataset.forget_split="forget_batch_${i}" \
        paths.output_dir=${EVAL_DIR_CUR}
        
    # Extraer métricas de L_n
    METRICS_FILE="${EVAL_DIR_CUR}/metrics.json"
    
    # 3. Evaluar la unión de lotes anteriores (D_prev) si i > 1
    if [ $i -gt 1 ]; then
        echo "========================================================="
        echo "Iteración $i / 5: Evaluación Continua (Fase 3) - D_prev ($i)"
        echo "========================================================="
        
        EVAL_DIR_PREV="${CURRENT_RUN_DIR}/eval_prev_${i}"
        
        python src/eval.py \
            experiment=eval/sequential_eval \
            task_name="eval_prev_${i}" \
            model.adapter_path=${NEW_MODEL} \
            dataset.forget_split="forget_prev_${i}" \
            paths.output_dir=${EVAL_DIR_PREV}
            
        # Reasignamos METRICS_FILE al D_prev para verificar colapso global
        METRICS_FILE="${EVAL_DIR_PREV}/metrics.json"
    fi
    
    # 4. Alerta de Colapso Catastrófico (Caída drástica en la fluidez)
    if [ -f "$METRICS_FILE" ]; then
        python3 -c "
import json
import sys
import os

metrics_file = '$METRICS_FILE'
try:
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Buscamos la métrica de gibberish (fluidez)
    gibberish_score = None
    for k, v in metrics.items():
        if 'gibberish' in k.lower():
            gibberish_score = v
            break
            
    if gibberish_score is not None:
        # Asumimos que un score elevado (>0.5) indica alta probabilidad de gibberish
        if gibberish_score > 0.5:
            print('\n\033[31m[ALERT] ¡Colapso Catastrófico Detectado en M_' + str($i) + '! Caída drástica en la fluidez (Gibberish > 0.5): {:.4f}\033[0m\n'.format(gibberish_score))
except Exception as e:
    print('Warning al leer métricas de fluidez:', e)
"
    fi
    
    # Actualizar PREV_MODEL para la próxima iteración
    PREV_MODEL=$NEW_MODEL
done

echo "========================================================="
echo "¡Pipeline de Desaprendizaje Secuencial Completado Exitosamente!"
echo "========================================================="
