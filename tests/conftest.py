"""Run the tests without torch or demucs.

Only the parts of torch that voicextract's pure helpers touch are provided,
backed by numpy. The model itself is never loaded.
"""
import contextlib
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



class FakeTensor(np.ndarray):
    """numpy array with the handful of torch.Tensor methods the code calls."""

    def dim(self):
        return self.ndim

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.tensor = lambda data: np.asarray(data).view(FakeTensor)
    torch.linspace = lambda a, b, n: np.linspace(a, b, n)
    torch.cat = lambda arrays, dim=0: np.concatenate(arrays, axis=dim).view(FakeTensor)
    torch.zeros_like = np.zeros_like
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch

for name, attrs in {
    "demucs": {},
    "demucs.pretrained": {"get_model": None},
    "demucs.apply": {"apply_model": None},
    "demucs.audio": {"AudioFile": None},
}.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
