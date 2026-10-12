"""Combine benchmark_latency_*.py results into a markdown table (plain Python, no dependencies).

    python compare_latency.py latency/gr00t.json latency/drifting.json --out latency/comparison.md
ManiFeel runs at 10 Hz and executes 8 of the 16 predicted steps per call: one call has an 800 ms budget.
"""

import argparse
import json

CONTROL_PERIOD_MS = 100  # ManiFeel / recording rate 10 Hz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-action-steps", type=int, default=8)
    args = parser.parse_args()
    runs = [json.load(open(p)) for p in args.results]
    base = runs[0]

    lines = [f"# Inference latency: {' vs. '.join(r['model'] for r in runs)}", "",
             f"GPU: {base['gpu']}. Median over {base['results'][0]['end_to_end']['n']} calls with different "
             "ManiFeel observations (wrist 256x256 + EEF state + task text) -> one 16-step action chunk. "
             "End to end = the full policy call as in the eval server (pre-processing, backbone, action head, "
             "post-processing); backbone and action head are timed separately (CUDA-synchronized).", ""]
    for i, b in enumerate(r["batch_size"] for r in base["results"]):
        lines += [f"## Batch size {b}" + (" (one robot)" if b == 1 else " (ManiFeel's parallel eval envs)"), "",
                  "| Model | End to end | Backbone | Action head | Other | Speed-up end to end | Speed-up head |",
                  "|---|---|---|---|---|---|---|"]
        ref = base["results"][i]
        for r in runs:
            row = r["results"][i]
            e2e, bb, head = (row[k]["median_ms"] for k in ("end_to_end", "backbone_ms", "action_head_ms"))
            lines.append(f"| {r['model']} | {e2e:.1f} ms | {bb:.1f} ms | {head:.1f} ms | {row['other_median_ms']:.1f} ms"
                         f" | {ref['end_to_end']['median_ms'] / e2e:.2f}x"
                         f" | {ref['action_head_ms']['median_ms'] / head:.2f}x |")
        lines.append("")
    lines += ["## Parameters", "", "| Model | Backbone | Action head |", "|---|---|---|"]
    lines += [f"| {r['model']} | {r['params_backbone_M']:.0f} M | {r['params_action_head_M']:.0f} M |" for r in runs]
    lines += ["", f"Control budget: {args.n_action_steps} steps x {CONTROL_PERIOD_MS} ms = "
              f"{args.n_action_steps * CONTROL_PERIOD_MS} ms per call at 10 Hz."]
    text = "\n".join(lines) + "\n"
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
