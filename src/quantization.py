"""int8 quantization of the U-Net (experiments/08_int8_quantization.md).

Two paths that share the SAME quantizer locations:

  - SIMULATION (`SimUNet`): "fake" quantization in floating point (Nagel et al. 2021, § 2.3; Krishnamoorthi 2018, § 2.4).
    Runs on GPU, accepts any bit width (W4, W6...), used for ablations, the sensitivity analysis and QAT
    (STE gradient, built into `torch.fake_quantize_*`).
  - REAL (`export_int8`): genuine PyTorch int8 kernels (`torch.ao.quantization`, eager mode, x86/fbgemm engine, CPU only),
    to measure the size on disk and the latency. ⚠ `torch.ao.quantization` is deprecated in PyTorch 2.10 (migration to
    torchao): it still works and the version is pinned by uv.lock.

Choices common to both paths (values and sources in the experiment page):
  - BatchNorm statically folded into the preceding convolution (Nagel § 2.3.1 and § 4.2: static folding is as good as the
    more complex variants when starting from a trained model). Checked: output difference ~1e-5.
  - Weights: symmetric, per channel for Conv2d (supported); per tensor for ConvTranspose2d in the REAL path, because
    `torch.ao` has no per-channel quantization for them ("Per channel weight observer is not supported yet for
    ConvTranspose{n}d"); the simulation can also do them per channel (`convt_per_channel=True`) to measure what that costs.
  - Activations: asymmetric (affine), per tensor (Nagel § 2.4.1 and § 2.4.2, Krishnamoorthi § 2.6).
  - LeakyReLU cannot be fused with the convolution (only ReLU can): a convolution output is quantized BEFORE and AFTER
    LeakyReLU, as `torch.ao` does and as Nagel § 2.3.2 foresees. ReLU (decoder) does not change the grid.
  - Concatenation of skip connections: requantization on a common grid (Nagel § 2.3.3, Krishnamoorthi § 3.2.1), one
    quantizer at the output of each `cat`.
  - Final sigmoid in floating point (the logit of the 1x1 is dequantized before).
  - `reduce_range=True`: activations on 7 bits (0..127). PyTorch's recommendation for CPUs without VNNI (AVX2 only, as here):
    the fbgemm kernels accumulate u8 x s8 in pairs in saturating int16.
"""
import copy
import dataclasses
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as aq
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .models.unet import UNet

SAMPLES_PER_BATCH = 100_000      # values kept per quantizer and per batch for the "percentile" and "mse" ranges
MAX_SAMPLES = 1_000_000


@dataclass
class QuantConfig:
    w_bits: int = 8
    a_bits: int = 8
    w_per_channel: bool = True        # Conv2d (encodeurs, 1x1 finale)
    convt_per_channel: bool = False   # ConvTranspose2d: False = like the real path (per tensor)
    w_range: str = "minmax"           # minmax | mse
    act_range: str = "minmax"         # minmax | percentile | mse
    percentile: float = 99.99
    reduce_range: bool = True         # activations on a_bits - 1 bits (see the header)
    quantize_input: bool = True       # False: the network input stays in floating point (the 1st convolution receives floats)
    quantize_weights: bool = True
    quantize_acts: bool = True

    def act_range_ints(self):
        bits = self.a_bits - 1 if self.reduce_range else self.a_bits
        return 0, 2 ** bits - 1

    def w_range_ints(self):
        return -(2 ** (self.w_bits - 1)), 2 ** (self.w_bits - 1) - 1


# ----------------------------------------------------------------------------------------------------------------------
# BatchNorm folding
# ----------------------------------------------------------------------------------------------------------------------

def fold_unet(unet: UNet) -> UNet:
    """Copy of the U-Net where each BatchNorm is folded into the (de)convolution that precedes it (eval mode: training
    statistics). The last deconvolution and the final 1x1 have no BatchNorm."""
    u = copy.deepcopy(unet).eval()
    for e in u.encoders:
        e.conv = fuse_conv_bn_eval(e.conv, e.bn)
        e.bn = nn.Identity()
    for d in u.decoders:
        if not d.return_logit:
            d.deconv = fuse_conv_bn_eval(d.deconv, d.bn, transpose=True)
            d.bn = nn.Identity()
    return u


# ----------------------------------------------------------------------------------------------------------------------
# Quantizers
# ----------------------------------------------------------------------------------------------------------------------

class ActQuantizer(nn.Module):
    """Per-tensor affine activation quantizer. Modes: off (identity) | observe (identity + statistics) |
    quant (fake quantization, frozen range) | qat (like quant, the range follows a moving average during training)."""

    def __init__(self, qmin, qmax, ema=0.01):
        super().__init__()
        self.qmin, self.qmax, self.ema = qmin, qmax, ema
        self.observer = aq.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=qmin, quant_max=qmax)
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))
        self.mode = "off"
        self._samples = []
        self._n_samples = 0

    def reset(self):
        self.observer.reset_min_max_vals()
        self._samples, self._n_samples = [], 0

    def set_range(self, lo, hi):
        """Sets the range [lo, hi] (never empty, contains 0 like any torch.ao observer) and derives the scale and zero point from it."""
        self.observer.min_val.fill_(float(lo))
        self.observer.max_val.fill_(float(hi))
        scale, zp = self.observer.calculate_qparams()
        self.scale.copy_(scale.to(self.scale.device).reshape(1))
        self.zero_point.copy_(zp.to(self.zero_point.device).reshape(1))

    def _collect(self, x):
        flat = x.reshape(-1)
        if flat.numel() > SAMPLES_PER_BATCH:
            flat = flat[torch.randint(0, flat.numel(), (SAMPLES_PER_BATCH,), device=flat.device)]
        if self._n_samples < MAX_SAMPLES:
            self._samples.append(flat.float().cpu())
            self._n_samples += flat.numel()

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode == "observe":
            self.observer(x.detach())
            self._collect(x.detach())
            return x
        if self.mode == "qat" and self.training:
            with torch.no_grad():
                lo, hi = x.min().clamp(max=0), x.max().clamp(min=0)
                new_lo = self.observer.min_val * (1 - self.ema) + lo * self.ema
                new_hi = self.observer.max_val * (1 - self.ema) + hi * self.ema
                self.observer.min_val.copy_(new_lo)
                self.observer.max_val.copy_(new_hi)
                scale, zp = self.observer.calculate_qparams()
                self.scale.copy_(scale.reshape(1))
                self.zero_point.copy_(zp.reshape(1))
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.qmin, self.qmax)

    @torch.no_grad()
    def finalize(self, method="minmax", percentile=99.99):
        """Computes the range from what was observed. minmax: extremes; percentile: quantiles (100-p, p); mse: range
        shrunk by a factor r that minimizes the quantization squared error on the sample (grid search,
        Nagel § 3.1)."""
        lo, hi = float(self.observer.min_val), float(self.observer.max_val)
        if method != "minmax" and self._samples:
            s = torch.cat(self._samples)
            if method == "percentile":
                q = torch.quantile(s, torch.tensor([(100 - percentile) / 100, percentile / 100]))
                lo, hi = min(float(q[0]), 0.0), max(float(q[1]), 0.0)
            elif method == "mse":
                best = (float("inf"), lo, hi)
                for r in torch.linspace(0.2, 1.0, 81):
                    a, b = lo * float(r), hi * float(r)
                    self.set_range(a, b)
                    err = (torch.fake_quantize_per_tensor_affine(s, self.scale.cpu(), self.zero_point.cpu(), self.qmin, self.qmax) - s).pow(2).mean()
                    if float(err) < best[0]:
                        best = (float(err), a, b)
                lo, hi = best[1], best[2]
            else:
                raise ValueError(f"unknown act_range={method!r} (minmax | percentile | mse)")
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        self.set_range(lo, hi)
        self._samples, self._n_samples = [], 0


class WeightQuantizer(nn.Module):
    """Symmetric weight quantizer, per channel or per tensor. `axis`: output-channel axis of the weight (0 for Conv2d,
    1 for ConvTranspose2d). Modes: off | quant (parameters frozen by `finalize`) | qat (recomputed at every pass: the weights
    move). Scale = max|w| / ((qmax - qmin) / 2), zero point 0: the formula of torch.ao's symmetric observers, which is
    also the one of the real conversion."""

    EPS = float(torch.finfo(torch.float32).eps)

    def __init__(self, bits, per_channel, axis, method="minmax"):
        super().__init__()
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.axis, self.per_channel, self.method = axis, per_channel, method
        self.mode = "off"
        self.scale, self.zero_point = None, None
        self.clip = None   # retained max |w| (per channel / tensor): the real export clips the weights to this value (MSE range)

    def _absmax(self, w):
        if self.per_channel:
            return w.detach().abs().movedim(self.axis, 0).reshape(w.shape[self.axis], -1).amax(dim=1)
        return w.detach().abs().amax().reshape(1)

    def _qparams(self, w):
        mx = self._absmax(w)
        if self.method == "mse":
            mx = self._mse_clip(w.detach(), mx)
        self.clip = mx.clone()
        scale = (mx / ((self.qmax - self.qmin) / 2)).clamp_min(self.EPS)
        return scale, torch.zeros_like(scale, dtype=torch.int32)

    def _mse_clip(self, w, mx):
        """Range |w| <= r x max|w| that minimizes the squared error, r searched in [0.3, 1] (independently per channel if needed)."""
        wd = w.movedim(self.axis, 0).reshape(w.shape[self.axis], -1) if self.per_channel else w.reshape(1, -1)
        mx = mx.clamp_min(1e-12).reshape(-1, 1)
        best_err, best_r = None, None
        for r in torch.linspace(0.3, 1.0, 36, device=w.device):
            s = (mx * r) / ((self.qmax - self.qmin) / 2)
            q = torch.clamp(torch.round(wd / s), self.qmin, self.qmax) * s
            err = (q - wd).pow(2).sum(dim=1, keepdim=True)
            if best_err is None:
                best_err, best_r = err, torch.full_like(err, float(r))
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_r = torch.where(better, torch.full_like(err, float(r)), best_r)
        return (mx * best_r).reshape(-1)

    @torch.no_grad()
    def finalize(self, w):
        self.scale, self.zero_point = self._qparams(w)

    def forward(self, w):
        if self.mode == "off":
            return w
        if self.mode == "qat":
            with torch.no_grad():
                self.scale, self.zero_point = self._qparams(w)
        scale = self.scale.to(w.device)   # plain attribute (not a buffer): follows the weight if the network changes device
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(w, scale, self.zero_point.to(w.device), self.axis, self.qmin, self.qmax)
        return torch.fake_quantize_per_tensor_affine(w, float(scale), 0, self.qmin, self.qmax)


# ----------------------------------------------------------------------------------------------------------------------
# Simulated network
# ----------------------------------------------------------------------------------------------------------------------

class SimUNet(nn.Module):
    """Folded U-Net (BatchNorm absorbed) with fake quantizers at the same places as the real int8 path.

    Activation quantizers (names): `in` (input), `enc{i}.conv` then `enc{i}.act` (before/after LeakyReLU), `dec{i}.conv`
    (deconvolution output; the decoder ReLU does not change the grid), `cat{i}` (concatenation), `final.conv` (logit).
    Weight quantizers: `enc{i}`, `dec{i}`, `final`."""

    def __init__(self, unet: UNet, cfg: QuantConfig = None):
        super().__init__()
        self.cfg = cfg or QuantConfig()
        self.unet = fold_unet(unet)
        for p in self.unet.parameters():
            p.requires_grad_(True)
        amin, amax = self.cfg.act_range_ints()
        depth = len(self.unet.encoders)
        names = ["in"] + [f"enc{i}.{k}" for i in range(depth) for k in ("conv", "act")] + \
                [f"dec{i}.conv" for i in range(depth)] + [f"cat{i}" for i in range(depth - 1)] + ["final.conv"]
        self.act_names = names
        self.aq = nn.ModuleDict({n.replace(".", "_"): ActQuantizer(amin, amax) for n in names})
        wb, pc, pt = self.cfg.w_bits, self.cfg.w_per_channel, self.cfg.convt_per_channel
        self.w_names = [f"enc{i}" for i in range(depth)] + [f"dec{i}" for i in range(depth)] + ["final"]
        self.wq = nn.ModuleDict()
        for i in range(depth):
            self.wq[f"enc{i}"] = WeightQuantizer(wb, pc, 0, self.cfg.w_range)
            self.wq[f"dec{i}"] = WeightQuantizer(wb, pt, 1, self.cfg.w_range)
        self.wq["final"] = WeightQuantizer(wb, pc, 0, self.cfg.w_range)

    # -- access
    def Q(self, name):
        return self.aq[name.replace(".", "_")]

    def set_mode(self, act=None, weights=None, only_act=None, only_w=None):
        """act ∈ {off, observe, quant, qat}; weights ∈ {off, quant, qat}. only_act / only_w: lists of names, the others go
        to `off` (per-quantizer sensitivity analysis)."""
        for n in self.act_names:
            if act is not None:
                on = only_act is None or n in only_act
                self.Q(n).mode = act if on else "off"
        for n in self.w_names:
            if weights is not None:
                on = only_w is None or n in only_w
                self.wq[n].mode = weights if on else "off"
        if not self.cfg.quantize_input:
            self.Q("in").mode = "off"

    @staticmethod
    def _qw(wq, w):
        return wq(w)

    def _layers(self):
        u = self.unet
        return ([(f"enc{i}", e.conv) for i, e in enumerate(u.encoders)] + [(f"dec{i}", d.deconv) for i, d in enumerate(u.decoders)]
                + [("final", u.final[0])])

    @torch.no_grad()
    def finalize_weights(self):
        for name, layer in self._layers():
            self.wq[name].finalize(layer.weight)

    @torch.no_grad()
    def finalize_acts(self):
        for n in self.act_names:
            self.Q(n).finalize(self.cfg.act_range, self.cfg.percentile)

    def forward(self, x):
        u = self.unet
        H, W = x.shape[-2:]
        m = 2 ** len(u.encoders)
        x = F.pad(x, (0, -W % m, 0, -H % m))
        x = self.Q("in")(x)
        skips = []
        for i, enc in enumerate(u.encoders):
            c = enc.conv
            z = F.conv2d(x, self.wq[f"enc{i}"](c.weight), c.bias, c.stride, c.padding)
            z = self.Q(f"enc{i}.conv")(z)
            x = self.Q(f"enc{i}.act")(enc.act(z))
            skips.append(x)
        skips = skips[:-1][::-1]
        for i, dec in enumerate(u.decoders):
            c = dec.deconv
            z = F.conv_transpose2d(x, self.wq[f"dec{i}"](c.weight), c.bias, c.stride, c.padding, c.output_padding)
            z = self.Q(f"dec{i}.conv")(z)
            if not dec.return_logit:
                z = dec.dropout(F.relu(z))
            x = z
            if i < len(skips):
                x = self.Q(f"cat{i}")(torch.cat([x, skips[i]], dim=1))
        c = u.final[0]
        x = self.Q("final.conv")(F.conv2d(x, self.wq["final"](c.weight), c.bias))
        return torch.sigmoid(x)[..., :H, :W]


@torch.no_grad()
def calibrate(sim: SimUNet, batches, quantize_weights=True, quantize_acts=True):
    """Runs the `batches` (network inputs, (B, C, F, T)) in floating point to observe the activation ranges, fixes the ranges and
    the weight parameters, then enables quantization. Observation is done on the FLOAT model (like torch.ao's `prepare`),
    not on the already quantized model."""
    sim.eval()
    for n in sim.act_names:
        sim.Q(n).reset()
    sim.set_mode(act="observe", weights="off")
    with torch.random.fork_rng():   # reproducible subsampling of the activations, without touching the caller's random state
        torch.manual_seed(0)
        for x in batches:
            sim(x)
    sim.finalize_acts()
    sim.finalize_weights()
    sim.set_mode(act="quant" if quantize_acts else "off", weights="quant" if quantize_weights else "off")
    return sim


# ----------------------------------------------------------------------------------------------------------------------
# Real int8 export (torch.ao, CPU)
# ----------------------------------------------------------------------------------------------------------------------

class Int8UNet(nn.Module):
    """Same graph as `SimUNet`, written for torch.ao eager quantization (QuantStub, FloatFunctional.cat)."""

    def __init__(self, folded: UNet):
        super().__init__()
        self.u = folded
        self.quant = aq.QuantStub()
        self.dequant = aq.DeQuantStub()
        self.cats = nn.ModuleList([nn.quantized.FloatFunctional() for _ in folded.decoders[:-1]])

    def forward(self, x):
        u = self.u
        H, W = x.shape[-2:]
        m = 2 ** len(u.encoders)
        x = self.quant(F.pad(x, (0, -W % m, 0, -H % m)))
        skips = []
        for enc in u.encoders:
            x = enc.act(enc.conv(x))
            skips.append(x)
        skips = skips[:-1][::-1]
        for i, dec in enumerate(u.decoders):
            x = dec.deconv(x)
            if not dec.return_logit:
                x = dec.dropout(dec.act(x))
            if i < len(skips):
                x = self.cats[i].cat([x, skips[i]], dim=1)
        x = self.dequant(u.final[0](x))
        return torch.sigmoid(x)[..., :H, :W]


def export_int8(sim: SimUNet, engine: str = "x86") -> Int8UNet:
    """Converts to a real int8 model (CPU) with the same ranges as the simulation (activation ranges injected into the
    observers; weights: minmax of the weight, possibly clipped to the MSE range, which gives the same grid). Reserved for
    W8A8 with the default quantizers."""
    cfg = sim.cfg
    if cfg.w_bits != 8 or cfg.a_bits != 8 or not (cfg.quantize_weights and cfg.quantize_acts and cfg.quantize_input):
        raise ValueError("real export: W8A8 with quantized input only")
    if cfg.convt_per_channel:
        raise ValueError("real export: ConvTranspose2d can only be quantized per tensor in torch.ao")
    torch.backends.quantized.engine = engine
    folded = copy.deepcopy(sim.unet).cpu().eval()
    # weights clipped to the retained range (no-op with minmax)
    with torch.no_grad():
        for name, layer in zip([n for n, _ in sim._layers()], [l for _, l in _layers_of(folded)]):
            q = sim.wq[name]
            if q.clip is not None:
                clip = q.clip.cpu()
                shape = [1] * layer.weight.dim()
                shape[q.axis] = -1
                layer.weight.copy_(torch.max(torch.min(layer.weight, clip.reshape(shape) if q.per_channel else clip),
                                             -(clip.reshape(shape) if q.per_channel else clip)))
    net = Int8UNet(folded).eval()
    amin, amax = cfg.act_range_ints()
    act = aq.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=amin, quant_max=amax)
    w_ten = aq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    w_chan = aq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    net.qconfig = aq.QConfig(activation=act, weight=w_ten)
    for enc in folded.encoders:
        enc.conv.qconfig = aq.QConfig(activation=act, weight=w_chan if cfg.w_per_channel else w_ten)
    folded.final[0].qconfig = aq.QConfig(activation=act, weight=w_chan if cfg.w_per_channel else w_ten)
    prepared = aq.prepare(net)
    # activation ranges: those of the simulation
    targets = {"in": prepared.quant}
    for i, enc in enumerate(prepared.u.encoders):
        targets[f"enc{i}.conv"], targets[f"enc{i}.act"] = enc.conv, enc.act
    for i, dec in enumerate(prepared.u.decoders):
        targets[f"dec{i}.conv"] = dec.deconv
    for i, c in enumerate(prepared.cats):
        targets[f"cat{i}"] = c
    targets["final.conv"] = prepared.u.final[0]
    for name, mod in targets.items():
        q = sim.Q(name)
        obs = mod.activation_post_process
        obs.min_val.fill_(float(q.observer.min_val))
        obs.max_val.fill_(float(q.observer.max_val))
    return aq.convert(prepared)


def _layers_of(unet):
    return ([(f"enc{i}", e.conv) for i, e in enumerate(unet.encoders)] + [(f"dec{i}", d.deconv) for i, d in enumerate(unet.decoders)]
            + [("final", unet.final[0])])


# ----------------------------------------------------------------------------------------------------------------------
# Measurements: size, latency
# ----------------------------------------------------------------------------------------------------------------------

def state_dict_bytes(module: nn.Module) -> int:
    """On-disk size of the serialized `state_dict` (what would be shipped)."""
    import io
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def latency_ms(model: nn.Module, x: torch.Tensor, threads: int, repeats: int = 7, warmup: int = 2):
    """Median (and min) time of a forward pass on CPU, in ms, with `threads` intra-op threads."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return ts[len(ts) // 2], ts[0]
    finally:
        torch.set_num_threads(prev)
