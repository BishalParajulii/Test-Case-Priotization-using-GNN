import os
import subprocess
import sys

from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.core.config import settings


def run_scale_gnn(experiment_id: int):
    db = SessionLocal()

    try:
        # 1️⃣ Fetch experiment
        experiment = db.query(Experiment).get(experiment_id)
        if not experiment:
            return

        dataset = db.query(Dataset).get(experiment.dataset_id)
        if not dataset:
            experiment.status = "FAILED"
            db.commit()
            return

        # 2️⃣ Mark RUNNING
        experiment.status = "RUNNING"
        db.commit()

        # 3️⃣ Paths (ABSOLUTE – very important)
        base_dir = os.getcwd()  # backend/
        data_path = os.path.join(base_dir, dataset.path)

        out_dir = os.path.join(
            base_dir,
            settings.OUTPUT_DIR,
            f"experiment_{experiment.id}"
        )
        os.makedirs(out_dir, exist_ok=True)

        params = experiment.parameters or {}

        threshold = params.get("threshold", 0.5)
        k_clusters = params.get("k_clusters", 10)
        latent_dim = params.get("latent_dim", 64)
        epochs = params.get("epochs", 50)
        topk = params.get("topk", 10)

        # 🔥 Critical: ensure scale_gnn is importable
        env = {
            **os.environ,
            "PYTHONPATH": base_dir,
        }

        python_exec = sys.executable  # same venv

        # =========================
        # 4️⃣ Fault-centric pruning
        # =========================
        subprocess.run(
            [
                python_exec, "-m", "scale_gnn.cli", "reduce",
                "--edges", f"{data_path}/coverage_edges.csv",
                "--history", f"{data_path}/history.csv",
                "--current", f"{data_path}/test_runs.csv",
                "--changed", f"{data_path}/changed_files.txt",
                "--out", f"{out_dir}/fault_prune",
                "--threshold", str(threshold),
            ],
            check=True,
            env=env
        )

        # =========================
        # 5️⃣ Spectral coarsening
        # =========================
        subprocess.run(
            [
                python_exec, "-m", "scale_gnn.cli", "spectral-coarsen",
                "--edges", f"{out_dir}/fault_prune/edges_fault_pruned.parquet",
                "--out", f"{out_dir}/spectral",
                "--k-clusters", str(k_clusters),
            ],
            check=True,
            env=env
        )

        # =========================
        # 6️⃣ Compression
        # =========================
        subprocess.run(
            [
                python_exec, "-m", "scale_gnn.cli", "compress",
                "--edges", f"{out_dir}/spectral/edges_spectral_coarsened.parquet",
                "--out", f"{out_dir}/compress",
                "--latent-dim", str(latent_dim),
                "--hidden-dim", "256",
                "--smooth-lambda", "1e-3",
                "--epochs", str(epochs),
                "--lr", "1e-3",
            ],
            check=True,
            env=env
        )

        # =========================
        # 7️⃣ Ranking
        # =========================
        subprocess.run(
            [
                python_exec, "-m", "scale_gnn.cli", "rank",
                "--edges", f"{out_dir}/spectral/edges_spectral_coarsened.parquet",
                "--history", f"{data_path}/history.csv",
                "--current", f"{data_path}/test_runs.csv",
                "--changed", f"{data_path}/changed_files.txt",
                "--embeddings", f"{out_dir}/compress",
                "--cluster-mapping", f"{out_dir}/spectral/spectral_cluster_mapping.csv",
                "--out", f"{out_dir}/rank",
                "--topk", str(topk),
                "--epochs", "100",
            ],
            check=True,
            env=env
        )

        # 8️⃣ Mark COMPLETED
        experiment.status = "COMPLETED"
        db.commit()

    except Exception as e:
        experiment.status = "FAILED"
        db.commit()
        print(f"[ERROR] Experiment {experiment_id} failed:", e)

    finally:
        db.close()
