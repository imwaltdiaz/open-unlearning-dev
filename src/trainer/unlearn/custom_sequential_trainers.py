import torch
from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_diff import GradDiff
from trainer.__init__ import TRAINER_REGISTRY, _register_trainer

class ULD(GradDiff):
    """Unlearning via Loss Difference."""
    def __init__(self, uld_lambda=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uld_lambda = uld_lambda

class SOUL(FinetuneTrainer):
    """Second-Order UnLearning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Stub for SOUL loss calculation
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

class WAGLE(FinetuneTrainer):
    """Weight-Routing Anti-Gradient Learning Engine."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Stub for WAGLE routing logic
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

class FLAT(FinetuneTrainer):
    """Forget Loss Only."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Stub for Forget Only logic
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# Register the classes in the registry
_register_trainer(ULD)
_register_trainer(SOUL)
_register_trainer(WAGLE)
_register_trainer(FLAT)
