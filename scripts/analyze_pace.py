"""Analyze HKJC race data to extract jockey pace archetypes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    races = pd.read_csv("data/races.csv")
    runs = pd.read_csv("data/runs.csv")
    return races, runs


def build_pace_profiles(races: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Build normalized pace profiles from sectional times.

    For each horse-run, compute the relative speed at each section
    (section_speed / average_speed). This normalizes across different
    race distances and conditions.
    """
    # Merge race distance info
    merged = runs.merge(races[["race_id", "distance"]], on="race_id")

    # Get sectional time columns that exist per horse
    time_cols = ["time1", "time2", "time3", "time4", "time5", "time6"]
    pos_cols = ["position_sec1", "position_sec2", "position_sec3",
                "position_sec4", "position_sec5", "position_sec6"]

    # Only use races with at least 3 sectional times (most common: 3-4 sections)
    merged["n_sections"] = merged[time_cols].notna().sum(axis=1)
    merged = merged[merged["n_sections"] >= 3].copy()

    profiles = []
    for _, row in merged.iterrows():
        # Get available sectional times
        times = []
        positions = []
        for tc, pc in zip(time_cols, pos_cols):
            if pd.notna(row[tc]) and row[tc] > 0:
                times.append(row[tc])
                positions.append(row[pc] if pd.notna(row[pc]) else np.nan)

        if len(times) < 3:
            continue

        # Compute speed per section (assuming equal section distances)
        # Speed = distance_per_section / time
        n = len(times)
        section_dist = row["distance"] / n
        speeds = [section_dist / t for t in times]
        avg_speed = sum(speeds) / len(speeds)

        if avg_speed < 1e-6:
            continue

        # Normalize: relative speed (1.0 = average)
        rel_speeds = [s / avg_speed for s in speeds]

        # Normalize to exactly 4 quartiles (early, mid-early, mid-late, late)
        # by interpolating to 4 evenly spaced points
        x_orig = np.linspace(0, 1, len(rel_speeds))
        x_quart = np.array([0.125, 0.375, 0.625, 0.875])  # quartile midpoints
        q_speeds = np.interp(x_quart, x_orig, rel_speeds)

        # Similarly for positions (normalize to fraction of field)
        valid_pos = [p for p in positions if not np.isnan(p)]
        if len(valid_pos) >= 3:
            # Get number of horses in this race
            race_horses = merged[merged["race_id"] == row["race_id"]].shape[0]
            if race_horses > 1:
                x_pos = np.linspace(0, 1, len(valid_pos))
                # Normalize position to [0, 1] where 0 = leading, 1 = last
                norm_pos = [(p - 1) / (race_horses - 1) for p in valid_pos]
                q_pos = np.interp(x_quart, x_pos, norm_pos)
            else:
                q_pos = [0.0, 0.0, 0.0, 0.0]
        else:
            continue

        profiles.append({
            "race_id": row["race_id"],
            "horse_id": row["horse_id"],
            "jockey_id": row["jockey_id"],
            "result": row["result"],
            "won": row["won"],
            "distance": row["distance"],
            # Quartile speeds (relative to average)
            "speed_q1": q_speeds[0],  # early
            "speed_q2": q_speeds[1],  # mid-early
            "speed_q3": q_speeds[2],  # mid-late
            "speed_q4": q_speeds[3],  # late
            # Quartile positions (0=leading, 1=last)
            "pos_q1": q_pos[0],
            "pos_q2": q_pos[1],
            "pos_q3": q_pos[2],
            "pos_q4": q_pos[3],
        })

    return pd.DataFrame(profiles)


def cluster_archetypes(profiles: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster pace profiles into archetypes using K-means."""
    feature_cols = [
        "speed_q1", "speed_q2", "speed_q3", "speed_q4",
        "pos_q1", "pos_q2", "pos_q3", "pos_q4",
    ]

    X = profiles[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = kmeans.fit_predict(X_scaled)

    return profiles


def label_archetypes(profiles: pd.DataFrame) -> dict:
    """Label clusters based on pace characteristics."""
    archetypes = {}

    for cluster_id in sorted(profiles["cluster"].unique()):
        cluster = profiles[profiles["cluster"] == cluster_id]

        avg = {
            "speed_q1": cluster["speed_q1"].mean(),
            "speed_q2": cluster["speed_q2"].mean(),
            "speed_q3": cluster["speed_q3"].mean(),
            "speed_q4": cluster["speed_q4"].mean(),
            "pos_q1": cluster["pos_q1"].mean(),
            "pos_q2": cluster["pos_q2"].mean(),
            "pos_q3": cluster["pos_q3"].mean(),
            "pos_q4": cluster["pos_q4"].mean(),
        }

        # Determine archetype based on position pattern
        early_pos = avg["pos_q1"]
        late_pos = avg["pos_q4"]
        pos_change = late_pos - early_pos  # negative = moves forward

        if early_pos < 0.3:
            label = "front_runner"
        elif early_pos > 0.6 and pos_change < -0.1:
            label = "closer"
        elif early_pos < 0.5 and pos_change < -0.05:
            label = "stalker"
        else:
            label = "presser"

        # Avoid duplicate labels
        existing_labels = [a["label"] for a in archetypes.values()]
        if label in existing_labels:
            label = f"{label}_{cluster_id}"

        win_rate = cluster["won"].mean()
        count = len(cluster)

        archetypes[str(cluster_id)] = {
            "label": label,
            "count": int(count),
            "win_rate": round(float(win_rate), 4),
            "pace_profile": {
                "speed_q1": round(avg["speed_q1"], 4),
                "speed_q2": round(avg["speed_q2"], 4),
                "speed_q3": round(avg["speed_q3"], 4),
                "speed_q4": round(avg["speed_q4"], 4),
            },
            "position_profile": {
                "pos_q1": round(avg["pos_q1"], 4),
                "pos_q2": round(avg["pos_q2"], 4),
                "pos_q3": round(avg["pos_q3"], 4),
                "pos_q4": round(avg["pos_q4"], 4),
            },
        }

    return archetypes


def main():
    print("Loading HKJC data...")
    races, runs = load_data()
    print(f"  Races: {len(races)}, Runs: {len(runs)}")

    print("\nBuilding pace profiles...")
    profiles = build_pace_profiles(races, runs)
    print(f"  Valid profiles: {len(profiles)}")

    print("\nClustering into 4 archetypes...")
    profiles = cluster_archetypes(profiles, n_clusters=4)

    archetypes = label_archetypes(profiles)

    print("\n" + "=" * 70)
    print("JOCKEY PACE ARCHETYPES")
    print("=" * 70)

    for cid, arch in sorted(archetypes.items(), key=lambda x: x[1]["label"]):
        print(f"\n  [{arch['label'].upper()}] (cluster {cid})")
        print(f"  Count: {arch['count']} runs ({arch['count']/len(profiles)*100:.1f}%)")
        print(f"  Win rate: {arch['win_rate']:.1%}")
        pp = arch["pace_profile"]
        print(f"  Speed curve:    Q1={pp['speed_q1']:.3f}  Q2={pp['speed_q2']:.3f}  Q3={pp['speed_q3']:.3f}  Q4={pp['speed_q4']:.3f}")
        rp = arch["position_profile"]
        print(f"  Position curve: Q1={rp['pos_q1']:.3f}  Q2={rp['pos_q2']:.3f}  Q3={rp['pos_q3']:.3f}  Q4={rp['pos_q4']:.3f}")

    # Save
    output = {
        "source": "HKJC Racing Dataset (Kaggle: gdaley/hkracing)",
        "total_profiles": len(profiles),
        "archetypes": archetypes,
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/archetypes.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to data/archetypes.json")


if __name__ == "__main__":
    main()
