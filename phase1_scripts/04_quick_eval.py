# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig

from evals import get_evaluators
from model import get_model
from trainer.utils import seed_everything


def log(msg):
    print(msg, flush=True)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Quick evaluation Fase 1 usando metricas del framework."""
    seed_everything(cfg.seed)

    model_cfg = cfg.model
    adapter_path = model_cfg.get("adapter_path", None)
    if adapter_path is None:
        log("adapter_path no esta configurado en el modelo. Usare el modelo base.")

    model, tokenizer = get_model(model_cfg)
    evaluators = get_evaluators(cfg.eval)

    for _, evaluator in evaluators.items():
        evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            template_args=model_cfg.template_args,
        )


if __name__ == "__main__":
    main()
