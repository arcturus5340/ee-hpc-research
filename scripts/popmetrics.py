import numpy as np
import pandas as pd


NUM_SHARED_METRICS = 7

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def emulate_pop_metrics_calculations(csv_path: str):
    df = read_csv(csv_path)

    required = ["rank","thr","socket",
                "uc_ct","uc_pt","uc_tt",
                "omc_ct","omc_pt","omc_tt",
                "ooc_ct","ooc_pt","ooc_tt",
                "total_runtime",
                "uc_ce","uc_pe","uc_te",
                "omc_ce","omc_pe","omc_te",
                "ooc_ce","ooc_pe","ooc_te",
                "total_energy"]

    df = df.copy()
    for c in required:
        if c not in df.columns and c.startswith("omc_"):
            df[c] = -np.inf
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ranks = sorted(df["rank"].unique())
    per_rank = {}

    for r in ranks:
        rdf = df[df["rank"] == r]
        num_threads = len(rdf)

        local_uc_avg_e = np.zeros(NUM_SHARED_METRICS)
        local_uc_max_e = np.zeros(NUM_SHARED_METRICS)

        for _, row in rdf.iterrows():
            curr_uc_e = row["uc_te"]
            curr_oot_e = row["ooc_te"]

            if curr_uc_e > local_uc_max_e[0]:
                local_uc_max_e[0] = curr_uc_e

            if curr_oot_e > local_uc_max_e[2]:
                local_uc_max_e[2] = curr_oot_e

            local_uc_avg_e[0] += curr_uc_e
            local_uc_avg_e[2] += curr_oot_e

        local_uc_max_e[1] = rdf["omc_pe"].iloc[0]
        local_uc_avg_e[1] = rdf["omc_pe"].iloc[0]

        local_uc_max_e[3] = rdf["uc_pe"].iloc[0]
        local_uc_avg_e[3] = rdf["uc_pe"].iloc[0]

        local_uc_max_e[4] = rdf["ooc_pe"].iloc[0]
        local_uc_avg_e[4] = rdf["ooc_pe"].iloc[0]

        local_uc_max_e[5] = local_uc_avg_e[3] - local_uc_avg_e[4]
        local_uc_avg_e[5] = (local_uc_avg_e[3] - local_uc_avg_e[4]) * num_threads

        local_uc_max_e[6] = local_uc_max_e[0]
        local_uc_avg_e[6] = local_uc_max_e[0] * num_threads

        localRuntimeRealEnergy = rdf["total_energy"].max()

        per_rank[r] = {
            "num_threads": num_threads,
            "socket": int(rdf["socket"].iloc[0]),
            "uc_avg_e": local_uc_avg_e.copy(),
            "uc_max_e": local_uc_max_e.copy(),
            "localRuntimeRealEnergy": localRuntimeRealEnergy
        }

    maxComputation = np.zeros(NUM_SHARED_METRICS)
    maxComputation.fill(-np.inf)

    sockets = sorted(df["socket"].unique())

    # calculate using rank 0 values
    first_socket = df.loc[df["rank"] == 0, "socket"].iloc[0]
    sockets = [first_socket] + [s for s in sockets if s != first_socket]
    
    num_sockets = len(sockets)

    socket_index = {s: i for i, s in enumerate(sockets)}
    socketComm_size = {s: 0 for s in sockets}

    totalRuntimeRealEnergy = np.zeros(num_sockets)

    socketMaxComputationEnergy = np.zeros((num_sockets, NUM_SHARED_METRICS))
    socketMaxComputationEnergy.fill(-np.inf)
    socketAvgComputationEnergy = np.zeros((num_sockets, NUM_SHARED_METRICS))

    total_threads = np.zeros(num_sockets)

    for r in ranks:
        s = per_rank[r]["socket"]
        idx = socket_index[s]
        total_threads[idx] += per_rank[r]["num_threads"]
        totalRuntimeRealEnergy[idx] = max(totalRuntimeRealEnergy[idx], per_rank[r]["localRuntimeRealEnergy"])
        socketMaxComputationEnergy[idx] = np.maximum(socketMaxComputationEnergy[idx], per_rank[r]["uc_max_e"])
        socketAvgComputationEnergy[idx] += per_rank[r]["uc_avg_e"]
        socketComm_size[s] += 1

    maxComputationEnergy = np.zeros((num_sockets, NUM_SHARED_METRICS))
    maxComputationEnergy.fill(-np.inf)
    avgComputationEnergy = np.zeros((num_sockets, NUM_SHARED_METRICS))

    maxComputationEnergy[:, :] = socketMaxComputationEnergy
    avgComputationEnergy[:, :] = socketAvgComputationEnergy

    for s in sockets:
        idx = socket_index[s]
        scount = socketComm_size[s]

        avgComputationEnergy[idx][1] = avgComputationEnergy[idx][1] / scount
        avgComputationEnergy[idx][3] = avgComputationEnergy[idx][3] / scount
        avgComputationEnergy[idx][4] = avgComputationEnergy[idx][4] / scount

        avgComputationEnergy[idx][0] = avgComputationEnergy[idx][0] / total_threads[idx]
        avgComputationEnergy[idx][2] = avgComputationEnergy[idx][2] / total_threads[idx]
        avgComputationEnergy[idx][5] = avgComputationEnergy[idx][5] / total_threads[idx]
        avgComputationEnergy[idx][6] = avgComputationEnergy[idx][6] / total_threads[idx]

    totalRuntimeIdealEnergy = df["uc_ce"].iloc[0]
    totalOutsideOMPIdealEnergy = df["ooc_ce"].iloc[0]

    for idx in range(num_sockets):
        avgComputationEnergy[idx][5] = avgComputationEnergy[idx][5] + totalRuntimeRealEnergy[idx]
        maxComputationEnergy[idx][5] = maxComputationEnergy[idx][5] + totalRuntimeRealEnergy[idx]

    CommE_e = np.zeros(num_sockets)
    TE_e = np.zeros(num_sockets)
    SerE_e = np.zeros(num_sockets)
    LB_e = np.zeros(num_sockets)
    PE_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        CommE_e[idx] = maxComputationEnergy[idx][0] / totalRuntimeRealEnergy[0]
        TE_e[idx] = totalRuntimeIdealEnergy / totalRuntimeRealEnergy[0]
        SerE_e[idx] = maxComputationEnergy[idx][0] / totalRuntimeIdealEnergy
        LB_e[idx] = avgComputationEnergy[idx][0] / maxComputationEnergy[idx][0]
        PE_e[idx] = LB_e[idx] * CommE_e[idx]

    mpiLB_e = np.zeros(num_sockets)
    ompLB_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        mpiLB_e[idx] = avgComputationEnergy[idx][6] / maxComputationEnergy[idx][0]
        ompLB_e[idx] = LB_e[idx] / mpiLB_e[idx]

    ompTE_e = np.zeros(num_sockets)
    mpiTE_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        ompTE_e[idx] = totalOutsideOMPIdealEnergy / totalRuntimeRealEnergy[0]
        mpiTE_e[idx] = TE_e[idx] / ompTE_e[idx]

    mpiSerE_e = np.zeros(num_sockets)
    ompSerE_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        mpiSerE_e[idx] = maxComputationEnergy[idx][3] / totalRuntimeIdealEnergy
        ompSerE_e[idx] = SerE_e[idx] / mpiSerE_e[idx]

    mpiCommE_e = np.zeros(num_sockets)
    ompCommE_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        mpiCommE_e[idx] = mpiSerE_e[idx] * mpiTE_e[idx]
        ompCommE_e[idx] = ompSerE_e[idx] * ompTE_e[idx]

    ompPE_e = np.zeros(num_sockets)
    mpiPE_e = np.zeros(num_sockets)

    for idx in range(num_sockets):
        ompPE_e[idx] = ompLB_e[idx] * ompCommE_e[idx]
        mpiPE_e[idx] = mpiLB_e[idx] * mpiCommE_e[idx]

    metrics = pd.DataFrame({
        "socket": sockets,
        "socket_rank_count": [socketComm_size[s] for s in sockets],
        "PE_e": PE_e,
        "LB_e": LB_e,
        "CommE_e": CommE_e,
        "SerE_e": SerE_e,
        "TE_e": TE_e,
        "mpiPE_e": mpiPE_e,
        "mpiLB_e": mpiLB_e,
        "mpiCommE_e": mpiCommE_e,
        "mpiSerE_e": mpiSerE_e,
        "mpiTE_e": mpiTE_e,
        "ompPE_e": ompPE_e,
        "ompLB_e": ompLB_e,
        "ompCommE_e": ompCommE_e,
        "ompSerE_e": ompSerE_e,
        "ompTE_e": ompTE_e,
    })

    return {
        "avgComputationEnergy": avgComputationEnergy,
        "maxComputationEnergy": maxComputationEnergy,
        "uc_avg_e_perrank": np.array([per_rank[r]["uc_avg_e"] for r in ranks]),
        "uc_max_e_perrank": np.array([per_rank[r]["uc_max_e"] for r in ranks]),
        "totalRuntimeRealEnergy": totalRuntimeRealEnergy,
        "totalRuntimeIdealEnergy": totalRuntimeIdealEnergy,
        "totalOutsideOMPIdealEnergy": totalOutsideOMPIdealEnergy,
        "metrics": metrics
    }


if __name__ == "__main__":
    sample_csv = "report.csv"
    res = emulate_pop_metrics_calculations(sample_csv)
    print("Result metrics per socket:")
    print(res["metrics"].to_string(index=False))
