"""
Métricas F1 Score para evaluación de desaprendizaje.

Implementa:
- f1_score: F1 score agregado y por ejemplo
- f1_less_than_01: Proporción de ejemplos con F1 < 0.1
"""

import logging
import numpy as np
from sklearn.metrics import f1_score as sklearn_f1
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals

logger = logging.getLogger("evaluator")


def _compute_f1_per_example(model, batch, tokenizer, **kwargs):
    """Compute F1 score for a batch of examples.
    
    Assumes batch contains 'labels' and model outputs logits.
    For generative models, this is a simplified F1 over token predictions.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
        logits = outputs.logits
        labels = batch['labels']
        
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute predictions
        predictions = np.argmax(shift_logits.cpu().numpy(), axis=-1)
        shift_labels_np = shift_labels.cpu().numpy()
        
        # Compute F1 per sequence in batch
        f1_scores = []
        for pred_seq, label_seq in zip(predictions, shift_labels_np):
            # Filter out ignore index (-100)
            valid_mask = label_seq != -100
            if valid_mask.sum() == 0:
                f1_scores.append(0.0)
                continue
            
            pred_valid = pred_seq[valid_mask]
            label_valid = label_seq[valid_mask]
            
            try:
                f1 = sklearn_f1(label_valid, pred_valid, average='weighted', zero_division=0)
            except:
                f1 = 0.0
            f1_scores.append(f1)
        
        return {'f1': np.array(f1_scores)}


@unlearning_metric(name="f1_score")
def f1_score_metric(model, **kwargs):
    """Compute F1 scores and return aggregated value along with per-index scores.
    
    This metric is framework-compatible:
    - Uses kwargs for data/collators/batch_size injection
    - Returns {"agg_value": ..., "value_by_index": {...}}
    """
    data = kwargs.get("data")
    collator = kwargs.get("collators")
    batch_size = kwargs.get("batch_size", 16)
    tokenizer = kwargs.get("tokenizer")
    
    if data is None:
        logger.warning("F1 metric: 'data' not provided in kwargs")
        return {"agg_value": 0.0, "value_by_index": {}}
    
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    
    fun_args = {"tokenizer": tokenizer}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        _compute_f1_per_example,
        fun_args,
        "Computing F1 scores"
    )
    
    f1_values = np.array(
        [
            evals.get('f1', 0.0)
            for evals in scores_by_index.values()
            if evals is not None and evals.get('f1') is not None
        ]
    )
    
    if len(f1_values) == 0:
        return {"agg_value": 0.0, "value_by_index": scores_by_index}
    
    return {
        "agg_value": float(np.mean(f1_values)),
        "value_by_index": scores_by_index
    }


@unlearning_metric(name="f1_less_than_01")
def f1_less_than_01_metric(model, **kwargs):
    """Compute proportion of examples with F1 < 0.1.
    
    This metric depends on f1_score being pre-computed.
    Uses kwargs["pre_compute"]["f1_score"] to access parent metric results.
    """
    pre_compute = kwargs.get("pre_compute", {})
    
    if "f1_score" not in pre_compute:
        logger.warning("f1_less_than_01: parent metric 'f1_score' not in pre_compute")
        return {"agg_value": 0.0}
    
    f1_results = pre_compute["f1_score"].get("value_by_index", {})
    
    if not f1_results:
        return {"agg_value": 0.0}
    
    count_below = sum(
        1 for result in f1_results.values()
        if result is not None and result.get("f1", 1.0) < 0.1
    )
    total = len(f1_results)
    
    ratio = count_below / total if total > 0 else 0.0
    
    return {"agg_value": float(ratio)}
