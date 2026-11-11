"""Residual adapters for frozen multivariate forecasters."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("full", "history_only", "no_history_direct", "concatenation", "history_condition_replacement")


class CalendarBridge(nn.Module):
    """Pass the current batch's time features to the backbone."""

    def __init__(self, backbone):
        super().__init__()
        self.network = backbone
        self.xmark = None
        self.future_mark = None

    def forward(self, x):
        return self.network(x, self.xmark, self.future_mark)


class ZeroHistoryDirect(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, h):
        return h.new_zeros((*h.shape[:-1], self.horizon))


class HOVER(nn.Module):
    """Refine a frozen forecast with history and forecast-conditioned residuals."""

    def __init__(self, backbone, seed, variant="full", horizon=96):
        super().__init__()
        if variant not in VARIANTS or horizon < 1:
            raise ValueError("Unsupported HOVER variant or horizon")
        if variant != "full" and horizon != 96:
            raise ValueError("The four formal mechanism controls are frozen at horizon 96")
        self.horizon = horizon
        self.variant = variant
        self.mode = {"full": "BILINEAR", "history_only": "HISTORY", "no_history_direct": "BILINEAR",
                     "concatenation": "CONCAT", "history_condition_replacement": "SURROGATE"}[variant]
        self.base = copy.deepcopy(CalendarBridge(backbone))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(seed)
        self.history = nn.Sequential(nn.Linear(336, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.head_h = nn.Linear(64, horizon)
        self.future = nn.Sequential(nn.Linear(horizon, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.hist_proj = nn.Linear(64, 32)
        self.future_proj = nn.Linear(32, 32, bias=False)
        self.norm = nn.LayerNorm(32)
        self.head_q = nn.Linear(32, horizon)
        # Zero residual heads make the initial prediction match the backbone.
        nn.init.zeros_(self.head_h.weight)
        nn.init.zeros_(self.head_h.bias)
        nn.init.zeros_(self.head_q.weight)
        nn.init.zeros_(self.head_q.bias)
        if variant == "history_only":
            for key in ("future", "hist_proj", "future_proj", "norm", "head_q"):
                delattr(self, key)
        if variant == "no_history_direct":
            # Remove the direct residual while keeping history available to FCRP.
            self.head_h = ZeroHistoryDirect(horizon)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone dropout and running statistics fixed during adapter training.
        self.base.eval()
        return self

    def components(self, x, xmark=None, future_mark=None):
        """Return the refined forecast, frozen forecast, and residual statistics.

        The input is [batch, 336, channels]; both forecasts are
        [batch, horizon, channels].
        """
        if x.ndim != 3 or x.shape[1] != 336:
            raise ValueError("HOVER expects [batch,336,channels]")
        self.base.xmark = xmark
        self.base.future_mark = future_mark
        with torch.no_grad():
            b = self.base(x)
        if b.shape != (x.shape[0], self.horizon, x.shape[2]):
            raise ValueError("Frozen backbone forecast horizon/channel mismatch")
        # Normalize each channel using only its observed history window.
        z = x.transpose(1, 2)
        mu = z.mean(-1, keepdim=True)
        sd = (z.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        h = self.history((z - mu) / sd)
        dh = self.head_h(h)  # HRP predicts a direct residual from history.
        if self.mode == "HISTORY":
            dq = torch.zeros_like(dh)
            q = h[:, :, :32] * 0
        else:
            # FCRP conditions on changes from the last observation in history-scale units.
            coord = (b.transpose(1, 2) - z[:, :, -1:]) / sd
            if self.mode == "SURROGATE":
                coord = (z[:, :, -self.horizon:] - z[:, :, -1:]) / sd
            a = self.future(coord)
            u = self.hist_proj(h)
            v = self.future_proj(a)
            # The "concatenation" control uses additive fusion of the projections.
            q = self.norm(F.gelu(u + v) if self.mode == "CONCAT" else torch.tanh(u) * torch.tanh(v))
            dq = self.head_q(q)
        # Rescale the joint residual to the frozen forecast's coordinates.
        prediction = b + ((dh + dq) * sd).transpose(1, 2)
        return prediction, b, {
            "history_delta_rms": (dh * sd).square().mean().sqrt(),
            "conditional_delta_rms": (dq * sd).square().mean().sqrt(),
            "interaction_rms": q.square().mean().sqrt(),
        }

    def forward(self, x, xmark=None, future_mark=None):
        return self.components(x, xmark, future_mark)[0]


class Simple(nn.Module):
    """Capacity control that maps the same normalized inputs through one MLP."""

    def __init__(self, backbone, seed, horizon=96):
        super().__init__()
        self.horizon = horizon
        self.base = copy.deepcopy(backbone)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(336 + horizon, 70), nn.GELU(), nn.Linear(70, 70), nn.GELU())
        self.head = nn.Linear(70, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def components(self, x, xmark=None, future_mark=None):
        with torch.no_grad():
            b = self.base(x, xmark, future_mark)
        if x.ndim != 3 or x.shape[1] != 336 or b.shape != (x.shape[0], self.horizon, x.shape[2]):
            raise ValueError("Simple history/horizon/channel mismatch")
        z = x.transpose(1, 2)
        mu = z.mean(-1, keepdim=True)
        sd = (z.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        coord = (b.transpose(1, 2) - z[:, :, -1:]) / sd
        delta = self.head(self.encoder(torch.cat([(z - mu) / sd, coord], -1))) * sd
        return b + delta.transpose(1, 2), b, {}

    def forward(self, x, xmark=None, future_mark=None):
        return self.components(x, xmark, future_mark)[0]


def adapter_loss(prediction, target, frozen, retention):
    """Penalize forecast error and deviation from the frozen prediction."""
    primary = (prediction - target).square().mean()
    preserve = (prediction - frozen).square().mean()
    return primary + retention * preserve, primary, preserve
