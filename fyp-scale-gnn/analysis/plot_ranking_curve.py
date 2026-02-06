import matplotlib.pyplot as plt
from analysis.config import FIGURE_DIR

def plot_ranking_curve(ranking):
    df = ranking.sort_values("rank")  # use the passed DataFrame

    plt.figure(figsize=(9, 5))
    plt.plot(df["rank"], df["risk_score"], linewidth=2)
    plt.xlabel("Test Execution Order")
    plt.ylabel("Predicted Fault Risk")
    plt.title("SCALE-GNN Test Prioritization Curve")
    plt.grid(True)

    path = f"{FIGURE_DIR}/ranking_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[✓] Ranking curve saved → {path}")
