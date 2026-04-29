from trainer.unlearn.grad_diff import GradDiff


class FLAT(GradDiff):
    def __init__(self, flat_temperature=1.0, flat_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_temperature = flat_temperature
        self.flat_alpha = flat_alpha
