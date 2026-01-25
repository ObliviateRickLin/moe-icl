"""
Checkpoint / run directory selection helpers.

This repo saves:
  - state.pt (dict with model_state_dict/optimizer_state_dict/train_step)
  - model_{step}.pt (raw model.state_dict(), typically at keep_every_steps)

Many plotting/eval scripts want to prefer a specific step (e.g., 300000) but
should fall back gracefully when a run hasn't finished yet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


def _list_model_ckpts(run_dir: Path) -> dict[int, Path]:
    ckpts: dict[int, Path] = {}
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = _MODEL_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        ckpts[step] = p
    return ckpts


def select_ckpt_path(run_dir: Path, prefer_step: Optional[int] = None) -> Tuple[Optional[Path], Optional[int]]:
    """
    Select a checkpoint file under run_dir.

    Priority:
      1) model_{prefer_step}.pt if prefer_step is given and exists
      2) max model_{step}.pt with step <= prefer_step (if any)
      3) max model_{step}.pt (if any)
      4) state.pt

    Returns (path, step). step is only set for model_{step}.pt.
    """
    ckpts = _list_model_ckpts(run_dir)
    if prefer_step is not None:
        p = ckpts.get(int(prefer_step))
        if p is not None and p.exists():
            return p, int(prefer_step)
        # fall back to the largest step <= prefer_step
        le = [s for s in ckpts.keys() if s <= int(prefer_step)]
        if le:
            s = max(le)
            return ckpts[s], s

    if ckpts:
        s = max(ckpts.keys())
        return ckpts[s], s

    state = run_dir / "state.pt"
    if state.exists():
        return state, None

    return None, None


def latest_run_dir(exp_dir: Path, prefer_step: Optional[int] = None) -> Optional[Path]:
    """
    Pick the latest run directory under an experiment directory.

    We look for subdirs containing config.yaml + some checkpoint (model_*.pt or state.pt).
    If prefer_step is provided, runs that have model_{prefer_step}.pt are ranked ahead.
    """
    if not exp_dir.exists():
        return None

    candidates: list[tuple[int, float, Path]] = []
    for cfg in exp_dir.rglob("config.yaml"):
        run_dir = cfg.parent
        ckpt_path, ckpt_step = select_ckpt_path(run_dir, prefer_step=prefer_step)
        if ckpt_path is None:
            continue
        has_prefer = 0
        if prefer_step is not None and ckpt_step == int(prefer_step):
            has_prefer = 1
        candidates.append((has_prefer, ckpt_path.stat().st_mtime, run_dir))

    if not candidates:
        return None
    # Prefer having the requested step; then use newest checkpoint mtime.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]

