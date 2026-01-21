import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from app.core.config import settings
from app.service.result_service import (
    load_ranking,
    load_top_tests,
    load_metrics
)


def generate_experiment_report(experiment_id: int) -> str:
    """
    Generates PDF report for an experiment
    """
    report_path = os.path.join(
        settings.REPORT_DIR,
        f"experiment_{experiment_id}_report.pdf"
    )

    os.makedirs(settings.REPORT_DIR, exist_ok=True)

    ranking = load_ranking(experiment_id)
    metrics = load_metrics(experiment_id)
    top_tests = load_top_tests(experiment_id)

    # -----------------------------
    # 1️⃣ Generate ranking curve image
    # -----------------------------
    img_path = os.path.join(
        settings.REPORT_DIR,
        f"experiment_{experiment_id}_curve.png"
    )

    ranks = [r["rank"] for r in ranking]
    risks = [float(r["risk_score"]) for r in ranking]

    # Normalize for visualization
    max_risk = max(risks)
    risks = [r / max_risk for r in risks]

    plt.figure(figsize=(6, 4))
    plt.plot(ranks, risks)
    plt.xlabel("Test Rank")
    plt.ylabel("Normalized Risk Score")
    plt.title("Test Prioritization Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    # -----------------------------
    # 2️⃣ Build PDF
    # -----------------------------
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Experiment Report #{experiment_id}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Top-K Prioritized Tests</b>", styles["Heading2"]))
    for t in top_tests:
        story.append(Paragraph(t, styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Metrics</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Ranking Curve</b>", styles["Heading2"]))
    story.append(Image(img_path, width=400, height=250))

    doc.build(story)

    return report_path
