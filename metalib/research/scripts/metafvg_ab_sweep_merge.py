"""
Merge sharded metafvg_ab_sweep.py outputs into one combined cache.

Usage:
    PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep_merge.py --slug m15_4h --shards 8
"""
import argparse
import glob
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="e.g. m15_4h or h4_1week")
    args = parser.parse_args()

    pattern = os.path.join(DATA_DIR, f"metafvg_ab_sweep_data_{args.slug}_shard*.pkl")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise SystemExit(f"No shard files matched {pattern}")

    merged = {}
    for path in shard_files:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        print(f"{os.path.basename(path)}: {len(shard)} results")
        merged.update(shard)

    out_path = os.path.join(DATA_DIR, f"metafvg_ab_sweep_data_{args.slug}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"\nMerged {len(merged)} total (symbol, config) results from {len(shard_files)} shards -> {out_path}")


if __name__ == "__main__":
    main()
