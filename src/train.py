import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)
    
    # [NUEVO] Forzar que el adaptador sea entrenable para ciclos secuenciales
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "train"):
        model.train()

        
    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # --- INYECCIÓN MANUAL DEL DATASET DE DESAPRENDIZAJE ---
    # Si estamos desaprendiendo y el dataset se aplanó, lo forzamos a la estructura WGA
    if mode == "unlearn" and "train" in data:
        train_data = data["train"]
        if not hasattr(train_data, 'forget'):
            print("\n[HACK] Forzando estructura ForgetRetainDataset...")
            try:
                from data import get_datasets
                from data.unlearn import ForgetRetainDataset
                
                forget_ds = get_datasets(data_cfg.forget, tokenizer=tokenizer, template_args=template_args)
                retain_ds = get_datasets(data_cfg.retain, tokenizer=tokenizer, template_args=template_args)
                
                # Si retorna diccionario (múltiples splits), sacamos el dataset real
                if isinstance(forget_ds, dict):
                    forget_ds = list(forget_ds.values())[0]
                if isinstance(retain_ds, dict):
                    retain_ds = list(retain_ds.values())[0]
                    
                data["train"] = ForgetRetainDataset(
                    forget=forget_ds,
                    retain=retain_ds,
                    anchor=data_cfg.get("anchor", "forget")
                )
                print("[HACK] Estructura reconstruida exitosamente.")
            except Exception as e:
                print(f"[HACK ERROR] No se pudo reconstruir el dataset: {e}")
    # --------------------------------------------------------

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    # --- HACK DE EMERGENCIA (Doble candado contra HuggingFace) ---
    from omegaconf import OmegaConf
    OmegaConf.set_struct(trainer_cfg, False)
    if "args" not in trainer_cfg:
        trainer_cfg.args = {}
    trainer_cfg.args.remove_unused_columns = False
    # -------------------------------------------------------------

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()