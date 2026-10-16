"""Quantification int8 du U-Net (experiments/08_int8_quantization.md).

Deux chemins qui partagent les MÊMES emplacements de quantificateurs :

  - SIMULATION (`SimUNet`) : quantification « fictive » en virgule flottante (Nagel et al. 2021, § 2.3 ; Krishnamoorthi 2018, § 2.4).
    Tourne sur GPU, accepte n'importe quelle largeur de bits (W4, W6...), sert aux ablations, à l'analyse de sensibilité et au QAT
    (gradient par STE, intégré à `torch.fake_quantize_*`).
  - RÉEL (`export_int8`) : vrais noyaux int8 de PyTorch (`torch.ao.quantization`, mode eager, moteur x86/fbgemm, CPU seulement),
    pour mesurer la taille sur disque et la latence. ⚠ `torch.ao.quantization` est déclaré obsolète dans PyTorch 2.10 (migration vers
    torchao) : il reste fonctionnel mais la version est figée par uv.lock.

Choix communs aux deux chemins (valeurs et sources dans la page d'expérience) :
  - BatchNorm repliée statiquement dans la convolution précédente (Nagel § 2.3.1 et § 4.2 : le repliement statique est aussi bon que
    les variantes plus complexes quand on part d'un modèle entraîné). Vérifié : écart de sortie ~1e-5.
  - Poids : symétriques, par canal pour les Conv2d (supporté) ; par tenseur pour les ConvTranspose2d dans le chemin RÉEL, car
    `torch.ao` n'a pas de quantification par canal pour elles (« Per channel weight observer is not supported yet for
    ConvTranspose{n}d ») ; la simulation peut aussi les faire par canal (`convt_per_channel=True`) pour mesurer ce que ça coûte.
  - Activations : asymétriques (affines), par tenseur (Nagel § 2.4.1 et § 2.4.2, Krishnamoorthi § 2.6).
  - LeakyReLU n'est pas fusionnable avec la convolution (seule ReLU l'est) : une sortie de convolution est quantifiée AVANT et APRÈS
    LeakyReLU, comme le fait `torch.ao` et comme le prévoit Nagel § 2.3.2. ReLU (décodeur) ne change pas la grille.
  - Concaténation des skip-connections : requantification sur une grille commune (Nagel § 2.3.3, Krishnamoorthi § 3.2.1), un
    quantificateur à la sortie de chaque `cat`.
  - Sigmoïde finale en virgule flottante (le logit de la 1x1 est déquantifié avant).
  - `reduce_range=True` : activations sur 7 bits (0..127). Recommandation de PyTorch pour les CPU sans VNNI (AVX2 seulement, comme ici) :
    les noyaux fbgemm accumulent u8 x s8 par paires en int16 saturant.
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

SAMPLES_PER_BATCH = 100_000      # valeurs conservées par quantificateur et par lot pour les plages « percentile » et « mse »
MAX_SAMPLES = 1_000_000


@dataclass
class QuantConfig:
    w_bits: int = 8
    a_bits: int = 8
    w_per_channel: bool = True        # Conv2d (encodeurs, 1x1 finale)
    convt_per_channel: bool = False   # ConvTranspose2d : False = comme le chemin réel (par tenseur)
    w_range: str = "minmax"           # minmax | mse
    act_range: str = "minmax"         # minmax | percentile | mse
    percentile: float = 99.99
    reduce_range: bool = True         # activations sur a_bits - 1 bits (voir l'en-tête)
    quantize_input: bool = True       # False : l'entrée du réseau reste en flottant (la 1re convolution reçoit des flottants)
    quantize_weights: bool = True
    quantize_acts: bool = True

    def act_range_ints(self):
        bits = self.a_bits - 1 if self.reduce_range else self.a_bits
        return 0, 2 ** bits - 1

    def w_range_ints(self):
        return -(2 ** (self.w_bits - 1)), 2 ** (self.w_bits - 1) - 1


# ----------------------------------------------------------------------------------------------------------------------
# Repliement de la BatchNorm
# ----------------------------------------------------------------------------------------------------------------------

def fold_unet(unet: UNet) -> UNet:
    """Copie du U-Net où chaque BatchNorm est repliée dans la (dé)convolution qui la précède (mode eval : statistiques de
    l'entraînement). La dernière déconvolution et la 1x1 finale n'ont pas de BatchNorm."""
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
# Quantificateurs
# ----------------------------------------------------------------------------------------------------------------------

class ActQuantizer(nn.Module):
    """Quantificateur d'activation affine par tenseur. Modes : off (identité) | observe (identité + statistiques) |
    quant (quantification fictive, plage figée) | qat (comme quant, la plage suit une moyenne mobile pendant l'entraînement)."""

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
        """Fixe la plage [lo, hi] (jamais vide, contient 0 comme tout observateur de torch.ao) et en déduit échelle et point zéro."""
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
        """Calcule la plage depuis ce qui a été observé. minmax : extrêmes ; percentile : quantiles (100-p, p) ; mse : plage
        réduite par un facteur r qui minimise l'erreur quadratique de quantification sur l'échantillon (recherche en grille,
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
                raise ValueError(f"act_range={method!r} inconnu (minmax | percentile | mse)")
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        self.set_range(lo, hi)
        self._samples, self._n_samples = [], 0


class WeightQuantizer(nn.Module):
    """Quantificateur de poids symétrique, par canal ou par tenseur. `axis` : axe des canaux de sortie du poids (0 pour Conv2d,
    1 pour ConvTranspose2d). Modes : off | quant (paramètres figés par `finalize`) | qat (recalculés à chaque passe : les poids
    bougent). Échelle = max|w| / ((qmax - qmin) / 2), point zéro 0 : la formule des observateurs symétriques de torch.ao, qui
    est aussi celle de la conversion réelle."""

    EPS = float(torch.finfo(torch.float32).eps)

    def __init__(self, bits, per_channel, axis, method="minmax"):
        super().__init__()
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.axis, self.per_channel, self.method = axis, per_channel, method
        self.mode = "off"
        self.scale, self.zero_point = None, None
        self.clip = None   # |w| maximal retenu (par canal / tenseur) : l'export réel écrête les poids à cette valeur (plage MSE)

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
        """Plage |w| <= r x max|w| qui minimise l'erreur quadratique, r cherché dans [0.3, 1] (indépendamment par canal si besoin)."""
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
        scale = self.scale.to(w.device)   # attribut simple (pas un buffer) : suit le poids si le réseau change de périphérique
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(w, scale, self.zero_point.to(w.device), self.axis, self.qmin, self.qmax)
        return torch.fake_quantize_per_tensor_affine(w, float(scale), 0, self.qmin, self.qmax)


# ----------------------------------------------------------------------------------------------------------------------
# Réseau simulé
# ----------------------------------------------------------------------------------------------------------------------

class SimUNet(nn.Module):
    """U-Net replié (BatchNorm absorbée) avec des quantificateurs fictifs aux mêmes endroits que le chemin int8 réel.

    Quantificateurs d'activation (noms) : `in` (entrée), `enc{i}.conv` puis `enc{i}.act` (avant/après LeakyReLU), `dec{i}.conv`
    (sortie de déconvolution ; le ReLU du décodeur ne change pas la grille), `cat{i}` (concaténation), `final.conv` (logit).
    Quantificateurs de poids : `enc{i}`, `dec{i}`, `final`."""

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

    # -- accès
    def Q(self, name):
        return self.aq[name.replace(".", "_")]

    def set_mode(self, act=None, weights=None, only_act=None, only_w=None):
        """act ∈ {off, observe, quant, qat} ; weights ∈ {off, quant, qat}. only_act / only_w : listes de noms, les autres passent
        en `off` (analyse de sensibilité par quantificateur)."""
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
    """Passe les `batches` (entrées du réseau, (B, C, F, T)) en flottant pour observer les plages d'activation, fixe les plages et
    les paramètres de poids, puis active la quantification. L'observation se fait sur le modèle FLOTTANT (comme `prepare` de
    torch.ao), pas sur le modèle déjà quantifié."""
    sim.eval()
    for n in sim.act_names:
        sim.Q(n).reset()
    sim.set_mode(act="observe", weights="off")
    with torch.random.fork_rng():   # sous-échantillonnage des activations reproductible, sans toucher à l'état aléatoire de l'appelant
        torch.manual_seed(0)
        for x in batches:
            sim(x)
    sim.finalize_acts()
    sim.finalize_weights()
    sim.set_mode(act="quant" if quantize_acts else "off", weights="quant" if quantize_weights else "off")
    return sim


# ----------------------------------------------------------------------------------------------------------------------
# Export int8 réel (torch.ao, CPU)
# ----------------------------------------------------------------------------------------------------------------------

class Int8UNet(nn.Module):
    """Même graphe que `SimUNet`, écrit pour la quantification eager de torch.ao (QuantStub, FloatFunctional.cat)."""

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
    """Convertit en vrai modèle int8 (CPU) avec les mêmes plages que la simulation (plages d'activation injectées dans les
    observateurs ; poids : minmax du poids éventuellement écrêté à la plage du MSE, ce qui donne la même grille). Réservé à
    W8A8 avec les quantificateurs par défaut."""
    cfg = sim.cfg
    if cfg.w_bits != 8 or cfg.a_bits != 8 or not (cfg.quantize_weights and cfg.quantize_acts and cfg.quantize_input):
        raise ValueError("export réel : W8A8 avec entrée quantifiée uniquement")
    if cfg.convt_per_channel:
        raise ValueError("export réel : les ConvTranspose2d ne se quantifient que par tenseur dans torch.ao")
    torch.backends.quantized.engine = engine
    folded = copy.deepcopy(sim.unet).cpu().eval()
    # poids écrêtés à la plage retenue (no-op en minmax)
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
    # plages d'activation : celles de la simulation
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
# Mesures : taille, latence
# ----------------------------------------------------------------------------------------------------------------------

def state_dict_bytes(module: nn.Module) -> int:
    """Taille sur disque du `state_dict` sérialisé (ce qu'on livrerait)."""
    import io
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def latency_ms(model: nn.Module, x: torch.Tensor, threads: int, repeats: int = 7, warmup: int = 2):
    """Médiane (et min) du temps d'une passe avant sur CPU, en ms, avec `threads` threads intra-op."""
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
