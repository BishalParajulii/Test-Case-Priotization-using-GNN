import matplotlib.pyplot as plt
from analysis.config import FIGURE_DIR

def plot_reduction(fault, spectral):
    stages = ["Original", "Fault Prune", "Spectral"]
    edges = [
        fault["orig_edges"].iloc[0],
        fault["red_edges"].iloc[0],
        spectral["red_edges"].iloc[0],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(stages, edges)
    plt.ylabel("Number of Edges")
    plt.title("Graph Reduction Across Stages")
    plt.grid(axis="y")

    path = f"{FIGURE_DIR}/reduction_ratio.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[✓] Reduction plot saved → {path}")
