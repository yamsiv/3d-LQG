"""Replot saved enriched FEM diagnostics with accurate support/error labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    directory=Path("results/phase2b/fem_enriched_audit")
    output=directory/"fem_boundary_corrected.png"
    if output.exists():raise FileExistsError(output)
    data=json.loads((directory/"audit.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for level in [3,4,5,6]:
        rows=[r for r in data["boundary_checks"] if np.isclose(r["alpha"],np.pi/2)
              and r["refinement"]==level and r["location"]=="corner"]
        axes[0].loglog([r["epsilon"] for r in rows],
                      [1+r["relative_error"] for r in rows],"o-",label=f"refinement {level}")
        rows=[r for r in data["analytic_checks"] if np.isclose(r["alpha"],np.pi/2)
              and r["refinement"]==level]
        axes[1].plot([r["location"] for r in rows],
                     [r["mean_value_relative_defect"] for r in rows],"o-",label=f"refinement {level}")
    axes[0].axhline(1,color="black",linestyle=":",label="Exact relative mode")
    axes[0].set(xlabel="Positive coordinates epsilon near cone ray",
                ylabel="Numerical angular mode / exact mode",
                title="I3 connected enrichment: positive but inaccurate near corner")
    axes[1].axhline(0,color="black",linestyle=":")
    axes[1].set(ylabel="Sphere-averaged h / center h - 1",
                title="Harmonic mean-value defect, radius = half face distance")
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output,dpi=180);plt.close(fig)
    print(output)


if __name__=="__main__":main()
