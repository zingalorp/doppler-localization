"""Figures for inspecting a search."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .physics import SPEED_OF_SOUND, doppler_factor  # noqa: E402


def spectrogram(ax, spec, t_range=None, f_range=None, vmax=12.0):
    hop, df = spec.hop, spec.df
    extent = [spec.t[0] - hop / 2, spec.t[-1] + hop / 2, spec.f[0] - df / 2, spec.f[-1] + df / 2]
    ax.imshow(spec.score, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=vmax,
              interpolation="nearest")
    if t_range:
        ax.set_xlim(*t_range)
    if f_range:
        ax.set_ylim(*f_range)
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("frequency (Hz)")


def model_curve(ax, spec, track, observer, t0, f0, t_range, c=SPEED_OF_SOUND, **kw):
    """Draw the curve `observer` would hear from `track` if the recording started at absolute `t0`."""
    t = spec.t[(spec.t >= t_range[0]) & (spec.t <= t_range[1])]
    ax.plot(t, f0 * doppler_factor(track, np.asarray(observer, dtype=float), t0 + t, c), **kw)


def score_map(ax, observers, scores, *, spacing, tracks=(), highlight=(), truth=None, best=None, title=""):
    obs = np.asarray(observers)
    shown = np.isfinite(scores)
    size = (72 * spacing / 1000 / max(np.ptp(obs[:, 0]) / 1000, 1e-9) * ax.figure.get_figwidth() * 0.6) ** 2
    for tr in tracks:
        ax.plot(tr.pos[:, 0] / 1000, tr.pos[:, 1] / 1000, color="0.8", lw=0.6, zorder=0)
    for tr in highlight:
        ax.plot(tr.pos[:, 0] / 1000, tr.pos[:, 1] / 1000, color="tab:blue", lw=1.2, zorder=1)
    sc = ax.scatter(obs[shown, 0] / 1000, obs[shown, 1] / 1000, c=scores[shown], s=max(size, 1.0), marker="s",
                    cmap="viridis", linewidths=0, zorder=2)
    if truth is not None:
        ax.plot(truth[0] / 1000, truth[1] / 1000, "*", ms=14, mfc="none", mec="red", mew=1.5, zorder=3,
                label="true position")
    if best is not None:
        ax.plot(best[0] / 1000, best[1] / 1000, "x", ms=10, color="white", mew=2, zorder=4)
        ax.plot(best[0] / 1000, best[1] / 1000, "x", ms=10, color="black", mew=1, zorder=4, label="best candidate")
    ax.set_xlim(obs[:, 0].min() / 1000, obs[:, 0].max() / 1000)
    ax.set_ylim(obs[:, 1].min() / 1000, obs[:, 1].max() / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title(title)
    if truth is not None or best is not None:
        ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="curve score", shrink=0.8)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
