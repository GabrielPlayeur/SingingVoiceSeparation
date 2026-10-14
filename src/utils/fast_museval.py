"""museval.evaluate (BSSEval v4) avec une estimation des filtres de distorsion en mémoire bornée.

Pourquoi : en v4, museval estime ses filtres de 512 coefficients sur le morceau ENTIER avec des FFT de taille
2**ceil(log2(T + 511)) (2**24 points pour un morceau de 3 à 6 min). En stéréo, cela représente 20 des 33 s par morceau
et un pic de 4,3 Go de RAM pour 210 s de signal (mesuré), plus du double au-delà de 380 s.

Ce que fait ce module : il remplace `_compute_reference_correlations` et `_compute_projection_filters` par des versions
qui calculent exactement les mêmes corrélations croisées, mais par blocs (overlap-save) : seuls les retards |m| < filters_len
sont nécessaires, donc des FFT de 2**16 points suffisent. Le reste de museval (projections par fenêtre de 1 s, SDR) est
inchangé. Les résultats sont identiques à l'arrondi flottant près ; la vérification est dans `__main__`.

Usage de vérification :
    python -m src.utils.fast_museval [musdb18/processed/test/<morceau>]   # compare au museval d'origine
"""
import contextlib
import itertools

import museval
import museval.metrics as metrics
import numpy as np
import scipy.fft
from scipy.linalg import toeplitz

BLOCK_FFT = 2 ** 16  # taille de FFT par bloc ; le bloc utile vaut BLOCK_FFT - 2 * (filters_len - 1) échantillons


def _lag_correlations(a_list, b_list, pairs, filters_len, fft_size=BLOCK_FFT):
    """r_p[m] = sum_n a[n + m] * b[n] pour m = 0..L-1 (tête) et m = -1..-(L-1) (queue), pour chaque couple p = (ia, ib).

    Même quantité que `real(ifft(fft(a) * conj(fft(b))))[m]` de museval, mais calculée par blocs de `fft_size`
    échantillons : le bloc de b est placé à l'offset L-1 d'un tampon de a élargi de L-1 échantillons de chaque côté,
    donc aucun repliement circulaire pour |m| <= L-1. Les produits spectraux sont sommés sur les blocs avant l'IFFT.
    Renvoie (tête, queue) de formes (n_pairs, L) et (n_pairs, L-1).
    """
    L = filters_len
    block = fft_size - 2 * (L - 1)
    T = len(a_list[0])
    n_freq = fft_size // 2 + 1
    acc = np.zeros((len(pairs), n_freq), dtype=np.complex128)
    a_buf = np.zeros((len(a_list), fft_size))
    b_buf = np.zeros((len(b_list), fft_size))
    for s in range(0, T, block):
        n = min(block, T - s)
        lo, hi = max(s - (L - 1), 0), min(s + block + (L - 1), T)
        a_buf[:] = 0.0
        b_buf[:] = 0.0
        for i, a in enumerate(a_list):
            a_buf[i, lo - (s - (L - 1)):hi - (s - (L - 1))] = a[lo:hi]
        for i, b in enumerate(b_list):
            b_buf[i, L - 1:L - 1 + n] = b[s:s + n]
        fa = scipy.fft.rfft(a_buf, axis=1)
        fb = scipy.fft.rfft(b_buf, axis=1)
        for p, (ia, ib) in enumerate(pairs):
            acc[p] += fa[ia] * np.conj(fb[ib])
    c = scipy.fft.irfft(acc, n=fft_size, axis=1)
    return c[:, :L], c[:, -1:-L:-1]


def _compute_reference_correlations(reference_sources, filters_len):
    """Remplace metrics._compute_reference_correlations. Renvoie (G, reference_sources) : le tableau temporel (nsrc, nsampl,
    nchan), sans copie, joue le rôle de `sf` ; les canaux sont lus en vues et convertis en float64 bloc par bloc."""
    (nsrc, nsampl, nchan) = reference_sources.shape
    rows = list(itertools.product(range(nsrc), range(nchan)))
    flat = [reference_sources[j, :, c] for j, c in rows]
    combos = list(itertools.combinations_with_replacement(range(len(rows)), 2))
    # museval : ssf = corr(sf[j, c2], sf[i, c1]) avec (i, c1) <= (j, c2) dans l'ordre des combinaisons
    pairs = [(q, p) for p, q in combos]
    head, tail = _lag_correlations(flat, flat, pairs, filters_len)
    G = np.zeros((nsrc, nsrc, nchan, nchan, filters_len, filters_len))
    for k, (p, q) in enumerate(combos):
        (i, c1), (j, c2) = rows[p], rows[q]
        ss = toeplitz(np.hstack((head[k][0], tail[k])), r=head[k])
        G[j, i, c2, c1] = ss
        G[i, j, c1, c2] = ss.T
    return G, reference_sources


def _compute_projection_filters(G, sf, estimated_source):
    """Remplace metrics._compute_projection_filters ; `sf` est ici le tableau de références temporel (cf. ci-dessus)."""
    eps = np.finfo(float).eps
    (nsampl, nchan) = estimated_source.shape
    if len(G.shape) == 4:  # une seule source de référence
        G = G[None, None, ...]
        sf = sf[None, ...]
    nsrc = G.shape[0]
    n_ref_chan = sf.shape[2]
    filters_len = G.shape[-1]

    refs = [sf[j, :, cj] for j in range(nsrc) for cj in range(n_ref_chan)]
    ests = [estimated_source[:, c] for c in range(nchan)]
    pairs = [(j * n_ref_chan + cj, c) for j in range(nsrc) for cj in range(n_ref_chan) for c in range(nchan)]
    head, tail = _lag_correlations(refs, ests, pairs, filters_len)

    D = np.zeros((nsrc, nchan, filters_len, nchan))
    for k, (ir, c) in enumerate(pairs):
        j, cj = divmod(ir, n_ref_chan)
        D[j, cj, :, c] = np.hstack((head[k][0], tail[k]))
    D = D.reshape(nsrc * nchan * filters_len, nchan)
    G = metrics._reshape_G(G)
    try:
        C = np.linalg.solve(G + eps * np.eye(G.shape[0]), D).reshape(nsrc, nchan, filters_len, nchan)
    except np.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(nsrc, nchan, filters_len, nchan)
    if nsrc == 1:
        C = C[0]
    return C


@contextlib.contextmanager
def low_memory_bsseval():
    """Active les versions par blocs pendant le `with` (patch de variables de module : pas thread-safe)."""
    saved = metrics._compute_reference_correlations, metrics._compute_projection_filters
    metrics._compute_reference_correlations = _compute_reference_correlations
    metrics._compute_projection_filters = _compute_projection_filters
    try:
        yield
    finally:
        metrics._compute_reference_correlations, metrics._compute_projection_filters = saved


def evaluate(references, estimates, win=44100, hop=44100):
    """Mêmes sorties que museval.evaluate (mode v4, padding=True) : (sdr, isr, sir, sar). Arguments de forme
    (nsrc, nsampl, nchan). Appelle `metrics.bss_eval` directement : `museval.evaluate` recopie les deux tableaux (np.array),
    soit 2 x 240 Mo pour un morceau stéréo de 5,6 min. Les tableaux ne sont pas modifiés."""
    if estimates.shape[1] > references.shape[1]:  # museval.pad_or_truncate
        estimates = estimates[:, :references.shape[1], :]
    elif estimates.shape[1] < references.shape[1]:
        estimates = np.pad(estimates, [(0, 0), (0, references.shape[1] - estimates.shape[1]), (0, 0)])
    with low_memory_bsseval():
        sdr, isr, sir, sar, _ = metrics.bss_eval(references, estimates, compute_permutation=False, window=win, hop=hop,
                                                 framewise_filters=False, bsseval_sources_version=False)
    return sdr, isr, sir, sar


if __name__ == "__main__":
    import resource
    import sys
    import time

    import soundfile as sf

    track = sys.argv[1] if len(sys.argv) > 1 else "musdb18/processed/test/AM Contra - Heart Peripheral"
    voc = sf.read(f"{track}/vocals.wav", dtype="float32", always_2d=True)[0]
    acc = sf.read(f"{track}/accompaniment.wav", dtype="float32", always_2d=True)[0]
    est = voc + 0.3 * np.random.default_rng(0).standard_normal(voc.shape).astype(np.float32) * np.abs(voc).mean()
    refs, ests = np.stack([voc, acc]), np.stack([est, acc + voc - est])
    print(f"{track} : {voc.shape[0] / 44100:.0f} s, {voc.shape[1]} canal(aux)")

    t = time.time()
    new = evaluate(refs, ests)
    print(f"par blocs : {time.time() - t:5.1f} s, pic RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} Go")
    t = time.time()
    ref = museval.evaluate(refs, ests, win=44100, hop=44100)
    print(f"museval   : {time.time() - t:5.1f} s")
    for name, x, y in zip(("SDR", "ISR", "SIR", "SAR"), new, ref):
        print(f"{name}: écart max {np.nanmax(np.abs(x - y)):.2e} dB, mêmes NaN : {np.array_equal(np.isnan(x), np.isnan(y))}")
