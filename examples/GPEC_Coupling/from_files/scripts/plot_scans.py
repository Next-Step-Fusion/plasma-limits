"""Plot complete scans and close-ups of the solved-point threshold brackets."""

from pathlib import Path
import argparse
import json
import os
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output.resolve()
os.environ.setdefault("MPLCONFIGDIR", str(out / "mpl_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = tomllib.loads((out / "results.toml").read_text())["results"]
summary = json.loads((out / "summary.json").read_text())
names = list(summary["datasets"])
fig, axes = plt.subplots(
    2, len(names), figsize=(6 * len(names), 8), layout="constrained", squeeze=False
)
for column, name in enumerate(names):
    group = [r for r in rows if r["dataset"] == name and r["status"] == "completed"]
    base = sorted(
        [r for r in group if r["preset"] == "baseline"], key=lambda r: r["multiplier"]
    )
    good = [r for r in base if r["screening_pass"]]
    bad = [r for r in base if not r["screening_pass"]]
    warning_points = {
        w["point"]
        for w in summary["datasets"][name]["solver_warnings"]
        if w["preset"] == "baseline"
        and any(
            "Free-boundary mode unstable" not in message for message in w["messages"]
        )
    }
    warned = [r for r in base if r["point"] in warning_points]
    result = summary["datasets"][name]["baseline"]
    for row_number, ax in enumerate(axes[:, column]):
        ax.plot(
            [r["beta_n"] for r in good],
            [r["delta_w"] for r in good],
            "o-",
            label="Baseline: metric checks pass",
            markersize=4,
        )
        if bad:
            ax.scatter(
                [r["beta_n"] for r in bad],
                [r["delta_w"] for r in bad],
                marker="x",
                c="gray",
                label="Failed metric check",
            )
        if warned:
            ax.scatter(
                [r["beta_n"] for r in warned],
                [r["delta_w"] for r in warned],
                s=85,
                facecolors="none",
                edgecolors="red",
                label="Internal-diagnostic warning",
            )
        for preset, marker in [("fine", "s"), ("core", "^"), ("riccati", "d")]:
            selected = [r for r in group if r["preset"] == preset]
            ax.scatter(
                [r["beta_n"] for r in selected],
                [r["delta_w"] for r in selected],
                marker=marker,
                label=preset.capitalize(),
                s=26,
            )
        if result["status"] == "bracketed":
            low, high = result["beta_n_bracket"]
            ax.axvspan(
                low, high, color="orange", alpha=0.15, label="Solved-point bracket"
            )
            ax.axvline(
                result["beta_n_estimate"],
                color="black",
                ls=":",
                label=f"Linear zero: {result['beta_n_estimate']:.3f}",
            )
            if row_number == 1:
                margin = 0.3 * (high - low)
                ax.set_xlim(low - margin, high + margin)
                endpoint_energies = [
                    r["delta_w"]
                    for r in group
                    if r["point"] in result["bracket_points"]
                ]
                lower, upper = min(endpoint_energies), max(endpoint_energies)
                padding = 0.2 * (upper - lower)
                ax.set_ylim(lower - padding, upper + padding)
        title = (
            "TokaMaker: shot 147131"
            if name == "tokamaker"
            else "NSFsim: existing DIII-D scenario"
        )
        ax.set(
            xlabel="Normalized beta βN (GPEC convention)",
            ylabel="Minimum δW (GPEC normalization)",
            title=title
            + (" — complete scan" if row_number == 0 else " — threshold detail"),
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(alpha=0.2)
        if row_number == 0:
            ax.legend(fontsize=7)
fig.suptitle(
    "Prepared-equilibrium n=1 no-wall scans; producer constraints differ", fontsize=12
)
figure_path = out / "beta_limits.png"
fig.savefig(figure_path, dpi=180)
print(figure_path)
