"""A run directory to look at the rater app with, made of synthetic subjects.

    uv run python tests/demo_run.py DEST [--subjects N] [--panel-px PX] [--small]

writes ``DEST/data`` (a subjects directory with an ``fsaverage`` next to the subjects) and
``DEST/run`` (a local run that went through inventory, metrics, selection and render, with
``protocol.yaml`` installed), and prints the run directory. The subjects are the ellipsoid
brains of ``tests/synth.py``: by default 1 mm volumes of 160^3 voxels with 2,562 vertices
per surface, rendered at 320 px, which is what a rater's screen gets; ``--small`` gives the
16^3 subjects of the test fixtures at 64 px in a few seconds. The first subject carries a
5 mm thickness patch, so the run has a cluster with a native centroid.

Nothing here is a test. The frontend is developed against such a run (its tiles, its
``render.json``) before the server exists, and the browser test builds its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_template import write_synthetic_template  # pyrefly: ignore[missing-import]
from synth_tierb import (  # pyrefly: ignore[missing-import]
    add_tier_a,
    add_tier_b,
    cohort_patches,
    synthetic_cohort,
)

from surfcensus.config import MetricsConfig, new_local_config
from surfcensus.protocol import install_protocol
from surfcensus.render.model import RenderConfig
from surfcensus.run import Run
from surfcensus.sources.local import local


def build_demo_run(
    dest: Path, *, n_subjects: int = 3, panel_px: int = 320, small: bool = False
) -> Path:
    """Write the data and the run under ``dest`` (which must not hold them yet); return the
    run directory."""
    data, run_dir = dest / "data", dest / "run"
    data.mkdir(parents=True)
    template = write_synthetic_template(dest / "template")
    patch = cohort_patches()["medial"]
    if small:
        synthetic_cohort(data, n_subjects, template, seed=4, bad={0: patch})
    else:
        for position in range(n_subjects):
            subject = write_synthetic_subject(
                data,
                name=f"sub-{position:03d}",
                subdivisions=4,
                # the midline moves from subject to subject, so no two have the same files
                midline_x=-4.0 + 1.5 * position,
            )
            add_tier_b(
                subject,
                template,
                warp=0.05 + 0.04 * position,
                patch=patch if position == 0 else None,
                noise_seed=4000 + position,
            )
            add_tier_a(subject)
    (data / "fsaverage").symlink_to(template.path, target_is_directory=True)
    reference_n = max(2, n_subjects)
    config = new_local_config(
        data,
        render=RenderConfig(panel_px=panel_px),
        metrics=MetricsConfig(min_reference_n=reference_n, min_vertex_n=min(2, reference_n)),
    )
    with Run(run_dir, local(data), config=config) as run:
        run.inventory()
        run.metrics()
        selection = pd.DataFrame(
            {
                "subject_uid": run.public().subject_uids(),
                "reason": "all",
                "stratum": "all",
                "inclusion_prob": 1.0,
                "seed": config.app.queue_seed,
            }
        )
        run.public().replace_selection(selection, ["all"])
        run.render(video=False)
    install_protocol(run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--panel-px", type=int, default=None)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args(argv)
    if args.dest.exists() and any(args.dest.iterdir()):
        parser.error(f"{args.dest} is not empty")
    panel_px = args.panel_px or (64 if args.small else 320)
    run_dir = build_demo_run(
        args.dest, n_subjects=args.subjects, panel_px=panel_px, small=args.small
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
