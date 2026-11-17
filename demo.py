"""Score three synthetic trajectories against one multimodal obstacle.

All positions and distances are in meters. This example evaluates pointwise
risk against a fixed uncertain obstacle; it is not a closed-loop experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from risk_validation import Disk, Gaussian2D, GaussianMixture2D, disk_reference, shared_space_mc


def example():
    obstacle = GaussianMixture2D((
        (0.65, Gaussian2D((3.0, -0.55), 0.32)),
        (0.35, Gaussian2D((3.0, 0.75), 0.42)),
    ))
    paths = {
        "Lower lane": tuple((6.0 * i / 40, -1.1) for i in range(41)),
        "Center lane": tuple((6.0 * i / 40, 0.0) for i in range(41)),
        "Upper lane": tuple((6.0 * i / 40, 1.1) for i in range(41)),
    }
    return obstacle, paths, 0.35


def evaluate(seed: int, samples: int):
    obstacle, paths, radius = example()
    disks = tuple(Disk(point, radius) for points in paths.values() for point in points)
    estimates = shared_space_mc((obstacle,), disks, n_samples=samples, seed=seed)
    result = {
        "scenario": "synthetic static multimodal obstacle",
        "seed": seed,
        "sample_count": samples,
        "collision_radius_m": radius,
        "obstacle": asdict(obstacle),
        "risk_definition": "pointwise collision probability, not whole-trajectory probability",
        "trajectories": {},
    }
    offset = 0
    for name, points in paths.items():
        references = [disk_reference(obstacle, Disk(point, radius), panels=512) for point in points]
        finer = [disk_reference(obstacle, Disk(point, radius), panels=1024) for point in points]
        if max(abs(a - b) for a, b in zip(references, finer)) > 1e-8:
            raise ArithmeticError("Reference integration has not converged.")
        mc = estimates.disks[offset:offset + len(points)]
        result["trajectories"][name] = {
            "points_m": points,
            "reference_risk": finer,
            "mc_box_risk": [estimate.rectangle_probabilities[0] for estimate in mc],
            "mc_conditional_risk": [
                None if estimate.conditional_probabilities is None
                else estimate.conditional_probabilities[0] for estimate in mc
            ],
            "disk_hit_counts": [estimate.hit_count for estimate in mc],
            "peak_pointwise_risk": max(finer),
            "mean_pointwise_risk": sum(finer) / len(finer),
        }
        offset += len(points)
    return result


def render(result, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np

    obstacle, paths, radius = example()
    x, y = np.linspace(0, 6, 81), np.linspace(-2, 2, 51)
    grid = np.array([
        [disk_reference(obstacle, Disk((float(px), float(py)), radius), panels=64)
         for px in x] for py in y
    ])
    colors = ("#0072B2", "#D55E00", "#009E73")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.5), constrained_layout=True)
    image = axes[0].pcolormesh(x, y, grid, shading="auto", cmap="magma", vmin=0)
    fig.colorbar(image, ax=axes[0], label="Pointwise collision probability", pad=0.02)
    for (name, points), color in zip(paths.items(), colors):
        xy = np.asarray(points)
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=2.0, label=name)
        trace = result["trajectories"][name]
        axes[1].plot(xy[:, 0], trace["reference_risk"], color=color, lw=2, label=name)
        axes[1].scatter(xy[::4, 0], trace["mc_box_risk"][::4], color=color, s=19, marker="x")
    axes[0].add_patch(Circle((0.6, -1.1), radius, fill=False, color="#56B4E9", lw=1.7))
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="One obstacle, two possible locations")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set(xlabel="Position along trajectory: x (m)", ylabel="Collision probability",
                title="Disk integration vs. shared Monte Carlo")
    axes[1].text(0.98, 0.95, "Lines: numerical reference\nCrosses: Monte Carlo estimate",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", fontsize=9)
    fig.savefig(output / "risk_demo.png", dpi=180, facecolor="white")
    fig.savefig(output / "risk_demo.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--no-plot", action="store_true", help="Run using only the Python standard library.")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.seed, args.samples)
    (args.output_dir / "risk_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_plot:
        render(result, args.output_dir)
    for name, trace in result["trajectories"].items():
        print(f"{name}: peak pointwise risk = {trace['peak_pointwise_risk']:.6f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
