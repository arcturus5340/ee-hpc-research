vethae5timport os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import itertools
import popmetrics


def flatten_metrics(metrics):
    metric_map = {
        "PE_e": "epe",
        "LB_e": "elb",
        "CommE_e": "ece",
        "SerE_e": "ese",
        "TE_e": "ete",
        "mpiPE_e": "empe",
        "mpiLB_e": "emlb",
        "mpiCommE_e": "emce",
        "mpiSerE_e": "emse",
        "mpiTE_e": "emte",
        "ompPE_e": "eope",
        "ompLB_e": "eolb",
        "ompCommE_e": "eoce",
        "ompSerE_e": "eose",
        "ompTE_e": "eote",
    }

    flat = {}
    for idx, row in metrics.iterrows():
        for col, prefix in metric_map.items():
            flat[f"{prefix}{idx}"] = row[col]

    return pd.DataFrame(flat, index=[0])


if __name__ == "__main__":
    base_dir = Path("../data/multisocket/icon_reports")
    summary_path = Path("../data/multisocket/icon_summary_annotated_before_recalculation.csv")
    output_path = Path("../data/multisocket/icon_summary_annotated.csv")
    
    summary_df = pd.read_csv(summary_path)
    
    int_cols = ["nodes", "taskspernode", "threadspertask", "iconsteps", "cpufreq", "slurm_job_id"]
    for col in int_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0).astype(int)
    
    updated_rows = []
    pattern = re.compile(
        r"(?P<tasks>\d+)_ranks_(?P<threads>\d+)_threads_(?P<freq>\d+)_hz_(?P<steps>\d+)_steps\.csv"
    )

    for file in sorted(base_dir.glob("*.csv")):
        match = pattern.match(file.name)
        if not match:
            print(f"[WARNING] Skipped file: {file.name}")
            continue
    
        tasks = int(match["tasks"])
        threads = int(match["threads"])
        freq = int(match["freq"])
        steps = int(match["steps"])
    
        mask = (
            (summary_df["taskspernode"] == tasks)
            & (summary_df["threadspertask"] == threads)
            & (summary_df["cpufreq"] == freq)
            & (summary_df["iconsteps"] == steps)
        )
    
        if not mask.any():
            print(f"[WARNING] No match found: {file.name}")
            continue
    
        row_idx = summary_df[mask].index[0]
        res = popmetrics.emulate_pop_metrics_calculations(base_dir / file.name)
        new_values = flatten_metrics(res["metrics"]).iloc[0]

        for col in new_values.keys():
            if col in summary_df.columns and col in new_values:
                summary_df.at[row_idx, col] = new_values[col]

        updated_rows.append(summary_df.loc[row_idx])
    
    updated_df = pd.DataFrame(updated_rows)
        
    for col in int_cols:
        if col in updated_df.columns:
            updated_df[col] = pd.to_numeric(updated_df[col], errors="coerce").fillna(0).astype(int)
    
    for col in updated_df.columns:
        if col not in int_cols:
            if pd.api.types.is_numeric_dtype(updated_df[col]):
                updated_df[col] = updated_df[col].round(3)
                
    sort_cols = ["taskspernode", "threadspertask", "iconsteps", "cpufreq"]
    updated_df = updated_df.sort_values(by=sort_cols, ascending=True)
    
    updated_df.to_csv(output_path, index=False)
    print("Done!")
