from trainer.unlearn.grad_diff import GradDiff


class SOUL(GradDiff):
    def __init__(self, soul_beta=1.0, soul_margin=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soul_beta = soul_beta
        self.soul_margin = soul_margin
