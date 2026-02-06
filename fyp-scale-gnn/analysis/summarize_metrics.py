def print_metrics(metrics):
    print("===== EXPERIMENT METRICS =====")

    # APFD
    apfd_value = metrics.get("apfd")
    if apfd_value is not None:
        print(f"APFD (SCALE-GNN): {apfd_value:.4f}")
    else:
        print("APFD key not found in metrics JSON.")

    # APFD Baseline
    apfd_baseline = metrics.get("apfd_baseline")
    if apfd_baseline is not None:
        print(f"APFD (Baseline): {apfd_baseline:.4f}")

    # Delta APFD
    delta_apfd = metrics.get("delta_apfd")
    if delta_apfd is not None:
        print(f"Delta APFD: {delta_apfd:.4f}")

    # Optional: Print all other metrics
    print("\nOther metrics:")
    for key, value in metrics.items():
        if key not in ["apfd", "apfd_baseline", "delta_apfd"]:
            print(f"{key}: {value}")
