"""Non-interactive batch driver: run the full pipeline on all 4 sectors sequentially.

Designed for overnight runs that produce the cross-sector headline comparison
without interactive prompts. Skips cached steps (preprocess, embed, build_graph)
when their outputs already exist. Per-sector logs go to results/<sector>/pipeline_log_<run_id>.txt.
A single sector failing does not abort the others.

Usage:
    uv run python run_all_sectors.py                 # defaults: K=200, W_dist=5, W_var=1
    uv run python run_all_sectors.py --K 100         # override hyperparameters
    uv run python run_all_sectors.py --only semicon  # run a single sector
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

SECTORS = [
    "semicon-1998-2012",
    "smartphone-1998-2012",
    "fcev-1998-2012",
    "antimono-1998-2012",
]
RAW_INPUTS_DIR = "raw_inputs"  # underscore — matches where the JSONLs actually live


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _run_step(name: str, cmd: list, log_handle) -> bool:
    """Run a subprocess; stream output to log; print only the step header to stdout."""
    header = f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> {name}\n    $ {' '.join(cmd)}\n"
    print(header, end="", flush=True)
    log_handle.write(header)
    log_handle.flush()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_handle.write(line)
    proc.wait()
    dur = _fmt_dur(time.time() - t0)
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    footer = f"    -> {status} in {dur}\n"
    print(footer, end="", flush=True)
    log_handle.write(footer)
    log_handle.flush()
    return proc.returncode == 0


def run_sector(sector: str, K: int, W_dist: float, W_var: float) -> bool:
    base_dir = f"results/{sector}"
    data_dir = f"{base_dir}/data"
    model_dir = f"{base_dir}/models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    jsonl_path = f"{RAW_INPUTS_DIR}/{sector}.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"  !! JSONL not found at {jsonl_path} — skipping sector")
        return False

    run_id = f"K{K}_Wd{W_dist}_Wv{W_var}"
    log_path = f"{base_dir}/pipeline_log_{run_id}.txt"

    processed_csv = f"{data_dir}/processed_patents.csv"
    embeddings_npy = f"{data_dir}/patent_embeddings.npy"
    graph_pt = f"{data_dir}/hetero_graph.pt"

    print(f"\n{'=' * 70}\n=== SECTOR: {sector}   run_id={run_id}\n{'=' * 70}")
    print(f"  Log: {log_path}")

    sector_t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"# Sector: {sector}\n# run_id: {run_id}\n# Started: {datetime.now().isoformat()}\n")

        # Step 1: preprocess (skip if already done)
        if os.path.exists(processed_csv):
            print("  [skip] preprocess — processed_patents.csv exists")
            log.write("\n[skip] preprocess — processed_patents.csv exists\n")
        else:
            if not _run_step("1/6 preprocess", [
                "python", "-u", "src/preprocess.py",
                "--input", jsonl_path, "--out_dir", data_dir,
            ], log):
                return False

        # Step 2: embed (skip if already done — this is the slow one, ~30 min on M1)
        if os.path.exists(embeddings_npy):
            print("  [skip] embed — patent_embeddings.npy exists")
            log.write("\n[skip] embed — patent_embeddings.npy exists\n")
        else:
            if not _run_step("2/6 embed (PatentSBERTa — slow)", [
                "python", "-u", "src/embed.py",
                "--input", processed_csv, "--out", embeddings_npy,
            ], log):
                return False

        # Step 3: build graph (skip if already done)
        if os.path.exists(graph_pt):
            print("  [skip] build_graph — hetero_graph.pt exists")
            log.write("\n[skip] build_graph — hetero_graph.pt exists\n")
        else:
            if not _run_step("3/6 build_graph", [
                "python", "-u", "src/build_graph.py", "--data_dir", data_dir,
            ], log):
                return False

        # Step 4: train (always)
        if not _run_step("4/6 train DMoN-DPR (200 epochs)", [
            "python", "-u", "src/train.py",
            "--data_dir", data_dir, "--model_dir", model_dir,
            "--K", str(K), "--W_dist", str(W_dist), "--W_var", str(W_var),
            "--run_id", run_id,
        ], log):
            return False

        # Step 5: analyze (always)
        if not _run_step("5/6 analyze clusters", [
            "python", "-u", "src/analyze.py",
            "--data_dir", data_dir, "--model_dir", model_dir,
            "--K", str(K), "--W_dist", str(W_dist), "--W_var", str(W_var),
            "--run_id", run_id,
        ], log):
            return False

        # Step 6: find triples (always)
        if not _run_step("6/6 find triples", [
            "python", "-u", "src/find_triples.py",
            "--data_dir", data_dir, "--run_id", run_id,
        ], log):
            return False

        dur = _fmt_dur(time.time() - sector_t0)
        log.write(f"\n# Finished: {datetime.now().isoformat()} (total {dur})\n")

    print(f"  DONE {sector} in {_fmt_dur(time.time() - sector_t0)}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--W_dist", type=float, default=5.0)
    p.add_argument("--W_var", type=float, default=1.0)
    p.add_argument("--only", type=str, default=None,
                   help="Run only one sector (substring match against sector names)")
    args = p.parse_args()

    sectors = SECTORS
    if args.only:
        sectors = [s for s in SECTORS if args.only in s]
        if not sectors:
            print(f"No sector matched '{args.only}'. Known sectors: {SECTORS}")
            sys.exit(1)

    print(f"Batch driver — sectors={sectors}  K={args.K}  W_dist={args.W_dist}  W_var={args.W_var}")
    print(f"Started at {datetime.now().isoformat()}")

    overall_t0 = time.time()
    results = {}
    for sector in sectors:
        try:
            results[sector] = run_sector(sector, args.K, args.W_dist, args.W_var)
        except KeyboardInterrupt:
            print(f"\n!! Interrupted during {sector}. Stopping.")
            break
        except Exception as e:
            print(f"\n!! Unexpected exception during {sector}: {e}")
            results[sector] = False

    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE in {_fmt_dur(time.time() - overall_t0)}")
    print("=" * 70)
    for sector, ok in results.items():
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {sector}")
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} sectors completed successfully.")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
