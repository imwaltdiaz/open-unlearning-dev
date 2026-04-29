from trainer.unlearn.grad_diff import GradDiff


class ULD(GradDiff):
    def __init__(self, uld_lambda=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uld_lambda = uld_lambda
