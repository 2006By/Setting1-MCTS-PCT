# -*- coding: utf-8 -*-
"""
Batch evaluate all joint model checkpoints in a directory and report the best one.

Example:
  python joint_eval_sweep.py ^
    --model-dir logs/joint_models/ablation_with_flat_gpu3 ^
    --test-dataset datasets/test.pt ^
    --continuous --shuffle --device 3 ^
    --save-csv logs/joint_models/ablation_with_flat_gpu3/eval_summary.csv
"""
import argparse
import concurrent.futures
import csv
import io
import multiprocessing
import os
import re
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

import givenData


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints in a model directory")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing *.pt model files")
    parser.add_argument("--test-dataset", type=str, default="datasets/test.pt", help="Test dataset path")
    parser.add_argument("--num-episodes", type=int, default=-1, help="Episodes per model (-1 means all)")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--no-cuda", action="store_true", help="Use CPU")
    parser.add_argument("--setting", type=int, default=1, help="Experiment setting")
    parser.add_argument("--window-size", type=int, default=5, help="Window size")
    parser.add_argument("--lnes", type=str, default="EMS", help="Leaf node expansion scheme")
    parser.add_argument("--internal-node-holder", type=int, default=100, help="Internal node holder")
    parser.add_argument("--leaf-node-holder", type=int, default=50, help="Leaf node holder")

    # Keep defaults aligned with your joint training/eval settings.
    parser.add_argument("--continuous", dest="continuous", action="store_true", help="Use continuous environment (default)")
    parser.add_argument("--discrete", dest="continuous", action="store_false", help="Use discrete environment")
    parser.set_defaults(continuous=True)

    parser.add_argument("--shuffle", dest="shuffle", action="store_true", help="Enable leaf-node shuffling (default)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable leaf-node shuffling")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--pattern", type=str, default=r"^(joint_model_\d+|joint_model_final|best_model)\.pt$",
                        help="Regex for model file names to evaluate")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Also include files matching *checkpoint*.pt if regex allows")
    parser.add_argument("--topk", type=int, default=10, help="Show top-k models")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for parallel evaluation (default: 1)")
    parser.add_argument("--save-csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--quiet-inner-eval", action="store_true",
                        help="Suppress verbose output from each single-model evaluation")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop sweep if one model fails")
    return parser.parse_args()


def build_eval_args(cli_args, model_path):
    args = SimpleNamespace()
    args.model_path = model_path
    args.test_dataset = cli_args.test_dataset
    args.num_episodes = cli_args.num_episodes
    args.setting = cli_args.setting
    args.continuous = cli_args.continuous
    args.no_cuda = cli_args.no_cuda
    args.device = "cpu" if cli_args.no_cuda else cli_args.device
    args.window_size = cli_args.window_size
    args.verbose = False

    # Joint model mode
    args.pure_pct = False

    # Model/env params expected by joint_eval
    args.embedding_size = 64
    args.hidden_size = 128
    args.gat_layer_num = 1
    args.internal_node_holder = cli_args.internal_node_holder
    args.leaf_node_holder = cli_args.leaf_node_holder
    args.lnes = cli_args.lnes
    args.shuffle = cli_args.shuffle

    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    args.id = "PctContinuous-v0" if args.continuous else "PctDiscrete-v0"
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    else:
        raise ValueError(f"Unsupported setting: {args.setting}")
    args.normFactor = 1.0 / np.max(args.container_size)
    return args


def extract_update(filename):
    m = re.match(r"joint_model_(\d+)\.pt$", filename)
    if m:
        return int(m.group(1))
    if filename == "best_model.pt":
        return -2
    if filename == "joint_model_final.pt":
        return -1
    return 10**12


def list_models(model_dir, pattern, include_checkpoints=False):
    regex = re.compile(pattern)
    names = []
    for name in os.listdir(model_dir):
        if not name.endswith(".pt"):
            continue
        if not include_checkpoints and "checkpoint" in name:
            continue
        if regex.match(name):
            names.append(name)
    names.sort(key=lambda n: (extract_update(n), n))
    return [os.path.join(model_dir, n) for n in names]


def evaluate_one(eval_fn, eval_args, quiet_inner_eval=False):
    if quiet_inner_eval:
        capture = io.StringIO()
        with redirect_stdout(capture):
            results = eval_fn(eval_args)
    else:
        results = eval_fn(eval_args)

    ratios = np.asarray(results["ratios"], dtype=np.float64)
    placed = np.asarray(results["placed"], dtype=np.float64)
    flatness = np.asarray(results.get("flatness_scores", []), dtype=np.float64)
    model_ms = np.asarray(results.get("model_infer_ms_per_step", []), dtype=np.float64)
    e2e_ms_ep = np.asarray(results.get("e2e_ms_per_box_episode", []), dtype=np.float64)

    mean_flatness = float(np.mean(flatness)) if flatness.size > 0 else 0.0
    std_flatness = float(np.std(flatness)) if flatness.size > 0 else 0.0
    var_flatness = float(np.var(flatness)) if flatness.size > 0 else 0.0
    mean_model_ms = float(np.mean(model_ms)) if model_ms.size > 0 else 0.0
    std_model_ms = float(np.std(model_ms)) if model_ms.size > 0 else 0.0
    var_model_ms = float(np.var(model_ms)) if model_ms.size > 0 else 0.0
    mean_e2e_ms = float(np.mean(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0
    std_e2e_ms = float(np.std(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0
    var_e2e_ms = float(np.var(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0
    return {
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "var_ratio": float(np.var(ratios)),
        "max_ratio": float(np.max(ratios)),
        "min_ratio": float(np.min(ratios)),
        "mean_placed": float(np.mean(placed)),
        "std_placed": float(np.std(placed)),
        "var_placed": float(np.var(placed)),
        "mean_flatness": mean_flatness,
        "std_flatness": std_flatness,
        "var_flatness": var_flatness,
        "mean_model_ms": mean_model_ms,
        "std_model_ms": std_model_ms,
        "var_model_ms": var_model_ms,
        "mean_e2e_ms": mean_e2e_ms,
        "std_e2e_ms": std_e2e_ms,
        "var_e2e_ms": var_e2e_ms,
        "episodes": int(len(ratios)),
    }


def save_csv(rows, path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "model_path",
                "mean_ratio",
                "std_ratio",
                "var_ratio",
                "max_ratio",
                "min_ratio",
                "mean_placed",
                "std_placed",
                "var_placed",
                "mean_flatness",
                "std_flatness",
                "var_flatness",
                "mean_model_ms",
                "std_model_ms",
                "var_model_ms",
                "mean_e2e_ms",
                "std_e2e_ms",
                "var_e2e_ms",
                "episodes",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = i
            writer.writerow(out)


def _build_worker_config(args):
    return {
        "test_dataset": args.test_dataset,
        "num_episodes": args.num_episodes,
        "setting": args.setting,
        "continuous": args.continuous,
        "no_cuda": args.no_cuda,
        "device": args.device,
        "window_size": args.window_size,
        "lnes": args.lnes,
        "internal_node_holder": args.internal_node_holder,
        "leaf_node_holder": args.leaf_node_holder,
        "shuffle": args.shuffle,
    }


def _evaluate_worker(payload):
    model_path = payload["model_path"]
    quiet_inner_eval = payload["quiet_inner_eval"]
    worker_config = payload["worker_config"]

    from joint_eval import evaluate as eval_fn
    from tools import registration_envs

    try:
        registration_envs()
    except Exception:
        # If envs are already registered in current process, continue.
        pass

    cli_args = SimpleNamespace(**worker_config)
    eval_args = build_eval_args(cli_args, model_path)

    t0 = time.time()
    metrics = evaluate_one(eval_fn, eval_args, quiet_inner_eval=quiet_inner_eval)
    metrics["elapsed_s"] = float(time.time() - t0)
    return {"model_path": model_path, **metrics}


def main():
    args = parse_args()

    # Delay import so that --help works even when runtime deps are not installed.
    from joint_eval import evaluate as eval_fn
    from tools import registration_envs
    try:
        registration_envs()
    except Exception:
        # If envs are already registered in current process, continue.
        pass

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir does not exist: {args.model_dir}")

    model_paths = list_models(args.model_dir, args.pattern, include_checkpoints=args.include_checkpoints)
    if not model_paths:
        raise RuntimeError("No model files matched. Please adjust --pattern or --include-checkpoints.")

    print(f"Found {len(model_paths)} models to evaluate in: {args.model_dir}")
    print(f"Dataset: {args.test_dataset} | continuous={args.continuous} | shuffle={args.shuffle}")
    print(f"Workers: {args.workers}")
    if args.workers > 1 and not args.no_cuda:
        print("Warning: --workers > 1 with CUDA uses the same GPU in multiple processes and may increase memory pressure.")
    print("-" * 80)

    all_rows = []
    best_row = None
    start = time.time()

    if args.workers <= 1:
        for idx, model_path in enumerate(model_paths, start=1):
            filename = os.path.basename(model_path)
            eval_args = build_eval_args(args, model_path)
            t0 = time.time()
            try:
                metrics = evaluate_one(eval_fn, eval_args, quiet_inner_eval=args.quiet_inner_eval)
                row = {"model_path": model_path, **metrics}
                all_rows.append(row)

                if best_row is None or row["mean_ratio"] > best_row["mean_ratio"]:
                    best_row = row

                cost = time.time() - t0
                print(
                    f"[{idx:03d}/{len(model_paths):03d}] {filename} "
                    f"| mean={row['mean_ratio']:.4f} var={row['var_ratio']:.6f} "
                    f"| episodes={row['episodes']} | {cost:.1f}s"
                )
            except Exception as e:
                print(f"[{idx:03d}/{len(model_paths):03d}] {filename} | FAILED: {e}")
                if args.stop_on_error:
                    raise
    else:
        max_workers = min(max(1, int(args.workers)), len(model_paths))
        mp_ctx = multiprocessing.get_context("spawn")
        worker_config = _build_worker_config(args)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            future_to_meta = {}
            for idx, model_path in enumerate(model_paths, start=1):
                payload = {
                    "model_path": model_path,
                    "quiet_inner_eval": args.quiet_inner_eval,
                    "worker_config": worker_config,
                }
                future = executor.submit(_evaluate_worker, payload)
                future_to_meta[future] = (idx, os.path.basename(model_path))

            for future in concurrent.futures.as_completed(future_to_meta):
                idx, filename = future_to_meta[future]
                try:
                    row = future.result()
                    all_rows.append(row)

                    if best_row is None or row["mean_ratio"] > best_row["mean_ratio"]:
                        best_row = row

                    print(
                        f"[{idx:03d}/{len(model_paths):03d}] {filename} "
                        f"| mean={row['mean_ratio']:.4f} var={row['var_ratio']:.6f} "
                        f"| episodes={row['episodes']} | {row.get('elapsed_s', 0.0):.1f}s"
                    )
                except Exception as e:
                    print(f"[{idx:03d}/{len(model_paths):03d}] {filename} | FAILED: {e}")
                    if args.stop_on_error:
                        for pending in future_to_meta:
                            pending.cancel()
                        raise

    if not all_rows:
        raise RuntimeError("All model evaluations failed.")

    ranked = sorted(all_rows, key=lambda x: x["mean_ratio"], reverse=True)
    topk = max(1, min(args.topk, len(ranked)))

    print("\n" + "=" * 80)
    print(f"Top {topk} Models (by mean ratio)")
    print("=" * 80)
    for i, row in enumerate(ranked[:topk], start=1):
        print(
            f"{i:02d}. {os.path.basename(row['model_path'])} "
            f"| mean={row['mean_ratio']:.4f} var={row['var_ratio']:.6f} "
            f"| max={row['max_ratio']:.4f} min={row['min_ratio']:.4f} "
            f"| placed={row['mean_placed']:.1f} | flat={row['mean_flatness']:.4f}"
        )

    print("\nBest model:")
    print(
        f"{best_row['model_path']} | mean={best_row['mean_ratio']:.4f} "
        f"var={best_row['var_ratio']:.6f} | episodes={best_row['episodes']}"
    )

    total = time.time() - start
    print(f"\nSweep finished in {total / 60.0:.1f} minutes.")

    if args.save_csv:
        save_csv(ranked, args.save_csv)
        print(f"Saved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
