"""A3: synthetic compositing-math tests (must pass before real frames).

Proves: C=aF+(1-a)B inversion recovers F with MAE<1e-4 on noise-free
alpha-sufficient pixels; luminance-preserving despill keeps |dY|<1e-6
pre-clip; despill does not touch achromatic pixels.
"""
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def synth(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.random((h, w, 3)) * 0.8 + 0.1
    B = np.zeros((h, w, 3))
    B[..., 1] = 0.35 + 0.1 * np.sin(np.linspace(0, 6, w))[None, :].repeat(h, 0)
    B[..., 0] = 0.08
    B[..., 2] = 0.10
    yy, xx = np.mgrid[:h, :w]
    a = np.clip(((xx - w / 2) + (yy - h / 2) * 0.3) / 12.0 + 0.5, 0, 1)
    C = a[..., None] * F + (1 - a[..., None]) * B
    return F, B, a, C


def recover(C, B, a, eps=1e-6):
    return (C - (1 - a[..., None]) * B) / np.maximum(a[..., None], eps)


def despill_luma(F, s, k):
    return F - (k * np.maximum(0.0, (F * s).sum(-1)))[..., None] * s[None, None, :]


def main():
    F, B, a, C = synth()
    Fh = recover(C, B, a)
    good = a > 0.7
    mae = float(np.abs(Fh[good] - F[good]).mean())
    print(f'inversion MAE (a>0.7): {mae:.2e}', flush=True)
    assert mae < 1e-4, mae

    s = np.array([-0.25, 0.9, -0.25])
    s = s - LUMA * (LUMA @ s) / (LUMA @ LUMA)  # chroma-plane projection
    s = s / np.linalg.norm(s)
    F2 = despill_luma(F, s, 0.5)
    dY = np.abs((F2 - F) @ LUMA).max()
    print(f'luma drift pre-clip: {dY:.2e}', flush=True)
    assert dY < 1e-6, dY

    gray = np.full((8, 8, 3), 0.4)
    g2 = despill_luma(gray, s, 1.0)
    assert float(np.abs(g2 - gray).max()) < 1e-9
    print('A3 synthetic math OK', flush=True)


if __name__ == '__main__':
    main()
