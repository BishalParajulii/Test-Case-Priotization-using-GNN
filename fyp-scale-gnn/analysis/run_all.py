from analysis.load_results import load_results
from analysis.plot_ranking_curve import plot_ranking_curve
from analysis.plot_reduction import plot_reduction
from analysis.summarize_metrics import print_metrics

if __name__ == "__main__":
    # 1️⃣ Load all experiment results
    ranking, metrics, fault, spectral = load_results()

    # 2️⃣ Print metrics
    print_metrics(metrics)

    # 3️⃣ Plot ranking curve
    plot_ranking_curve(ranking)

    # 4️⃣ Plot reduction graph
    plot_reduction(fault, spectral)
