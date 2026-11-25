"""Run the pre-registered SVI refit (docs/SVI_REFIT_PLAN.md) on the frozen snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import R, TICKER
from src.data import clean_chain, fetch_chain
from src.surface import build_surface
from src.svi import calibrate_surface

VARIANTS = {
    "A": dict(),
    "B": dict(objective="iv"),
    "C": dict(objective="iv", grid_starts=True),
    "D": dict(objective="iv", grid_starts=True, butterfly_weight=1.0),
}


def main() -> None:
    data = fetch_chain(TICKER, refresh=False)
    surf = build_surface(clean_chain(data["chain"]), spot=data["spot"], r=R, q=data["q"])
    out = {}
    for test, train_every in (("same_split", 1), ("sparse_every_3rd", 3)):
        out[test] = {}
        for name, opts in VARIANTS.items():
            fits = calibrate_surface(surf, spot=data["spot"], r=R, q=data["q"], train_every=train_every, **opts)
            svi, lin = fits["holdout_iv_rmse"], fits["linear_holdout_iv_rmse"]
            out[test][name] = dict(
                slices=len(fits),
                svi_mean_iv_rmse_volpts=round(100 * float(svi.mean()), 4),
                linear_mean_iv_rmse_volpts=round(100 * float(lin.mean()), 4),
                svi_wins=int(np.sum(svi < lin)),
                converged=int(fits["success"].sum()),
                butterfly_flags=int(fits["convexity_violations"].sum()),
            )
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/svi_refit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
