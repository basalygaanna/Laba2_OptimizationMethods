from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

def save_solution(x, bases, points, times, capacities, filename):
    rows = []
    for i, base in enumerate(bases):
        for j, point in enumerate(points):
            if x[i, j] > 1e-9:
                rows.append({
                    "base": base, "incident_point": point,
                    "teams": float(x[i, j]),
                    "travel_time_minutes_per_team": int(times[i, j]),
                    "arc_capacity_teams": int(capacities[i, j])
                })
    pd.DataFrame(rows).to_csv(ROOT / "results" / filename, index=False, encoding="utf-8-sig")

def save_network(x, bases, points, times, capacities, filename):
    """Readable bipartite network: active arcs are green and labeled flow/capacity."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    nb, ni = len(bases), len(points)
    fig, ax = plt.subplots(figsize=(13, 8))
    left_x, right_x = 0.08, 0.92
    base_y = [0.90 - i * (0.80 / max(1, nb - 1)) for i in range(nb)]
    point_y = [0.90 - j * (0.80 / max(1, ni - 1)) for j in range(ni)]
    pos = {bases[i]: (left_x, base_y[i]) for i in range(nb)}
    pos.update({points[j]: (right_x, point_y[j]) for j in range(ni)})

    # Draw all possible routes faintly; used routes are emphasized below.
    for i in range(nb):
        for j in range(ni):
            x0, y0 = pos[bases[i]]
            x1, y1 = pos[points[j]]
            flow = float(x[i, j])
            cap = int(capacities[i, j])
            if flow > 1e-9:
                color, lw, alpha = "#16803c", 2.8, 0.95
            else:
                color, lw, alpha = "#cbd5e1", 0.8, 0.65
            # A slight route-specific curve separates labels and makes endpoints clear.
            rad = (j - (ni - 1) / 2) * 0.018 + (i - (nb - 1) / 2) * 0.008
            patch = FancyArrowPatch(
                (x0 + 0.025, y0), (x1 - 0.025, y1),
                arrowstyle="-|>", mutation_scale=10,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw, color=color, alpha=alpha,
                zorder=1 if flow <= 1e-9 else 2
            )
            ax.add_patch(patch)
            if flow > 1e-9:
                # Put flow/capacity on the route itself; travel time is in the CSV.
                label_t = 0.34
                mx = x0 + (x1 - x0) * label_t
                my = y0 + (y1 - y0) * label_t + rad * 0.20
                ax.text(mx, my, f"{flow:g}/{cap}", fontsize=8, ha="center", va="center",
                        color="#14532d", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#86a98e", alpha=0.95))

    # Nodes and labels
    for name, (xx, yy) in pos.items():
        is_base = name in bases
        face = "#8ecae6" if is_base else "#ffb703"
        ax.scatter([xx], [yy], s=1600, color=face, edgecolors="#334155", linewidths=0.8, zorder=4)
        ax.text(xx, yy, name, ha="center", va="center", fontsize=9, zorder=6)

    ax.text(0.08, 1.02, "БАЗЫ", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(0.92, 1.02, "ТОЧКИ ЧС", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(0.5, -0.02, "Подпись на зелёной дуге: отправлено бригад / лимит дуги.\n"
            "Время движения по каждой использованной дуге указано в соответствующем CSV.",
            ha="center", va="top", fontsize=9)
    ax.set_title("Распределение аварийных бригад по сети", fontsize=14, pad=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.09, 1.08)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
