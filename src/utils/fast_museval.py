"""museval.evaluate (BSSEval v4) with the distortion filters estimated in bounded memory.

Why: in v4, museval estimates its 512-tap filters on the WHOLE track with FFTs of size
2**ceil(log2(T + 511)) (2**24 points for a 3 to 6 min track). In stereo, this accounts for 20 of the 33 s per track
and a peak of 4.3 GB of RAM for 210 s of signal (measured), more than twice that beyond 380 s.

What this module does: it replaces `_compute_reference_correlations` and `_compute_projection_filters` with versions
that compute exactly the same cross-correlations, but block by block (overlap-save): only the lags |m| < filters_len
are needed, so FFTs of 2**16 points are enough. The rest of museval (projections per 1 s window, SDR) is unchanged.
Results are identical up to floating-point rounding; the check is in `__main__`.

Check usage:
    python -m src.utils.fast_museval [musdb18/processed/test/<track>]   # compares with the original museval
"""
import contextlib
import itertools

import museval
import museval.metrics as metrics
import numpy as np
import scipy.fft
from scipy.linalg import toeplitz

BLOCK_FFT = 2 ** 16  # FFT size per block; the useful block is BLOCK_FFT - 2 * (filters_len - 1) samples


def _lag_correlations(a_list, b_list, pairs, filters_len, fft_size=BLOCK_FFT):
    """r_p[m] = sum_n a[n + m] * b[n] for m = 0..L-1 (head) and m = -1..-(L-1) (tail), for each pair p = (ia, ib).

    Same quantity as museval's `real(ifft(fft(a) * conj(fft(b))))[m]`, but computed in blocks of `fft_size`
    samples: the block of b is placed at offset L-1 of a buffer of a widened by L-1 samples on each side,
    so there is no circular wrap-around for |m| <= L-1. The spectral products are summed over the blocks before the IFFT.
    Returns (head, tail) of shapes (n_pairs, L) and (n_pairs, L-1).
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
    """Replaces metrics._compute_reference_correlations. Returns (G, reference_sources): the time-domain array (nsrc, nsampl,
    nchan), without copy, plays the role of `sf`; the channels are read as views and converted to float64 block by block."""
    (nsrc, nsampl, nchan) = reference_sources.shape
    rows = list(itertools.product(range(nsrc), range(nchan)))
    flat = [reference_sources[j, :, c] for j, c in rows]
    combos = list(itertools.combinations_with_replacement(range(len(rows)), 2))
    # museval: ssf = corr(sf[j, c2], sf[i, c1]) with (i, c1) <= (j, c2) in the order of the combinations
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
    """Replaces metrics._compute_projection_filters; `sf` is here the time-domain reference array (see above)."""
    eps = np.finfo(float).eps
    (nsampl, nchan) = estimated_source.shape
    if len(G.shape) == 4:  # a single reference source
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
    """Enables the block-wise versions inside the `with` (patches module variables: not thread-safe)."""
    saved = metrics._compute_reference_correlations, metrics._compute_projection_filters
    metrics._compute_reference_correlations = _compute_reference_correlations
    metrics._compute_projection_filters = _compute_projection_filters
    try:
        yield
    finally:
        metrics._compute_reference_correlations, metrics._compute_projection_filters = saved


def evaluate(references, estimates, win=44100, hop=44100):
    """Same outputs as museval.evaluate (v4 mode, padding=True): (sdr, isr, sir, sar). Arguments of shape
    (nsrc, nsampl, nchan). Calls `metrics.bss_eval` directly: `museval.evaluate` copies both arrays (np.array),
    i.e. 2 x 240 MB for a 5.6 min stereo track. The arrays are not modified."""
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
    print(f"{track}: {voc.shape[0] / 44100:.0f} s, {voc.shape[1]} channel(s)")

    t = time.time()
    new = evaluate(refs, ests)
    print(f"blockwise: {time.time() - t:5.1f} s, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    t = time.time()
    ref = museval.evaluate(refs, ests, win=44100, hop=44100)
    print(f"museval  : {time.time() - t:5.1f} s")
    for name, x, y in zip(("SDR", "ISR", "SIR", "SAR"), new, ref):
        print(f"{name}: max difference {np.nanmax(np.abs(x - y)):.2e} dB, same NaNs: {np.array_equal(np.isnan(x), np.isnan(y))}")
