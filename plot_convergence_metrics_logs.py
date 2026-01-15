import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def smooth(values: List[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return np.asarray(values, dtype=float)
    window = min(window, len(values))
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="valid")


def load_episode_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    file_path = run_dir / "convergence_metrics.json"
    if not file_path.exists():
        return [], {}
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  Skip {file_path} (invalid JSON: line {e.lineno}, col {e.colno})")
        return [], {}
    except Exception as e:  # catch-all to avoid breaking plotting when a file is corrupted
        print(f"⚠️  Skip {file_path} (error: {e})")
        return [], {}
    episodes = []
    metrics = {"mean_latency": [], "qos_scores": [], "peaked_mem": [], "total_revenue": []}
    for ep in data.get("episode_metrics", []):
        episodes.append(int(ep.get("episode", len(episodes) + 1)))
        final_info = ep.get("final_info", {})
        metrics["mean_latency"].append(float(final_info.get("mean_latency", 0.0)))
        metrics["qos_scores"].append(float(final_info.get("qos_scores", 0.0)))
        metrics["peaked_mem"].append(float(final_info.get("peaked_mem", 0.0)))
        metrics["total_revenue"].append(float(final_info.get("total_revenue", 0.0)))
    return episodes, metrics


def plot_run(run_name: str, episodes: List[int], metrics: Dict[str, List[float]], output_dir: Path, smooth_window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plots = [
        ("mean_latency", "Latency (s)", axes[0, 0]),
        ("qos_scores", "QoS Score", axes[0, 1]),
        ("peaked_mem", "Peak Memory", axes[1, 0]),
        ("total_revenue", "Revenue", axes[1, 1]),
    ]
    for key, ylabel, ax in plots:
        values = metrics.get(key, [])
        if not values:
            ax.set_title(f"{ylabel} (no data)")
            ax.axis("off")
            continue
        ax.plot(episodes, values, label="raw", alpha=0.45, linewidth=1.0)
        smoothed = smooth(values, smooth_window)
        if len(smoothed) > 0 and len(smoothed) != len(values):
            smoothed_x = episodes[smooth_window - 1 :]
        else:
            smoothed_x = episodes
        ax.plot(smoothed_x, smoothed, label=f"MA({smooth_window})" if smooth_window > 1 else "raw", linewidth=2.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{run_name} - {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(episodes) if episodes else 1)
        ax.legend()
    plt.tight_layout()
    save_path = output_dir / f"{run_name}_convergence.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_overlay(all_runs: Dict[str, Dict[str, List[float]]], episodes_map: Dict[str, List[int]], output_dir: Path, smooth_window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_keys = [
        ("mean_latency", "Latency (s)", "latency"),
        ("qos_scores", "QoS Score", "qos"),
        ("peaked_mem", "Peak Memory", "memory"),
        ("total_revenue", "Revenue", "revenue"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(all_runs))))
    for key, ylabel, short in metrics_keys:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, (run_name, metrics) in enumerate(all_runs.items()):
            values = metrics.get(key, [])
            episodes = episodes_map.get(run_name, [])
            if not values or not episodes:
                continue
            smoothed = smooth(values, smooth_window)
            if len(smoothed) > 0 and len(smoothed) != len(values):
                smoothed_x = episodes[smooth_window - 1 :]
            else:
                smoothed_x = episodes
            ax.plot(smoothed_x, smoothed, label=run_name, color=colors[idx], linewidth=2.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Convergence - {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(max(episodes_map.get(rn, [0])) for rn in all_runs.keys()) if all_runs else 1)
        ax.legend()
        plt.tight_layout()
        save_path = output_dir / f"convergence_{short}_overlay.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence metrics (latency, QoS, memory, revenue) per episode")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing log subfolders")
    parser.add_argument("--output_dir", type=str, default="plots/convergence", help="Directory to save plots")
    parser.add_argument("--smooth_window", type=int, default=5, help="Moving average window for smoothing")
    parser.add_argument("--overlay_only", action="store_true", help="Only create overlay plots")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    all_runs: Dict[str, Dict[str, List[float]]] = {}
    episodes_map: Dict[str, List[int]] = {}

    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        episodes, metrics = load_episode_metrics(run_dir)
        if not episodes:
            continue
        run_name = run_dir.name
        all_runs[run_name] = metrics
        episodes_map[run_name] = episodes
        if not args.overlay_only:
            plot_run(run_name, episodes, metrics, output_dir, args.smooth_window)

    if not all_runs:
        print("No convergence_metrics.json files found.")
        return

    plot_overlay(all_runs, episodes_map, output_dir, args.smooth_window)


if __name__ == "__main__":
    main()
