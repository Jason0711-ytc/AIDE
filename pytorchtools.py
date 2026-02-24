import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    This is a lightweight utility used by `run.py`.
    """

    def __init__(self, patience=7, verbose=False, delta=0.0, path=None, trace_func=print):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.delta = float(delta)
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model=None):
        score = -float(val_loss)

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f})."
            )
        self.val_loss_min = float(val_loss)

        if self.path and model is not None:
            try:
                torch.save(model.state_dict(), self.path)
            except Exception as exc:
                if self.verbose:
                    self.trace_func(f"Warning: failed to save checkpoint to {self.path}: {exc}")
