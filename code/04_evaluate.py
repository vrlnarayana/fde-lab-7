"""
04_evaluate.py  (LAB 7-C)
-------------------------
Runs the 50-question golden set against ALL THREE retrieval strategies on
Azure AI Search and reports recall@k, MRR@k, nDCG@k for each. Saves a results
table you can paste into ADR-2.

It also prints a per-category breakdown (single_hop / multi_hop / temporal) so
you can SEE where naive RAG fails and hybrid/re-ranking helps.

Run:
    python 04_evaluate.py            # default k=5
    python 04_evaluate.py --k 10
"""
import argparse, json, os
from collections import defaultdict

from azure_clients import get_vector_store
from eval_metrics import (recall_at_k, reciprocal_rank_at_k, ndcg_at_k, mean_ignore_nan)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.abspath(os.path.join(HERE, "..", "golden_set", "golden_set.json"))
OUT = os.path.abspath(os.path.join(HERE, "..", "golden_set", "results.csv"))

STRATEGIES = {"naive": "similarity", "hybrid": "hybrid", "reranked": "semantic_hybrid"}


def parent_ids(docs):
    seen, ordered = set(), []
    for d in docs:
        pid = d.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    k = args.k

    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    answerable = [q for q in gold if q["category"] != "negative"]

    # Pre-build one store per strategy (retrieval depth a bit larger than k
    # so collapsing chunks->parents still leaves k candidates).
    stores = {name: get_vector_store(search_type=st) for name, st in STRATEGIES.items()}

    rows = []
    per_cat = defaultdict(lambda: defaultdict(list))  # per_cat[strategy][category] = [recall...]
    summary = {}

    for name, store in stores.items():
        rec, mrr, ndcg = [], [], []
        for q in answerable:
            docs = store.similarity_search(q["question"], k=k + 5)
            got = parent_ids(docs)[:k]
            rel = q["relevant_doc_ids"]
            r = recall_at_k(got, rel, k)
            m = reciprocal_rank_at_k(got, rel, k)
            n = ndcg_at_k(got, rel, k)
            rec.append(r); mrr.append(m); ndcg.append(n)
            per_cat[name][q["category"]].append(r)
            rows.append({"strategy": name, "qid": q["qid"], "category": q["category"],
                         "recall": r, "rr": m, "ndcg": n, "retrieved": "|".join(got)})
        summary[name] = {
            f"recall@{k}": mean_ignore_nan(rec),
            f"MRR@{k}": mean_ignore_nan(mrr),
            f"nDCG@{k}": mean_ignore_nan(ndcg),
        }

    # Print overall table
    print(f"\n===== RETRIEVAL QUALITY (k={k}, {len(answerable)} answerable questions) =====")
    print(f"{'strategy':<10} {'recall':>8} {'MRR':>8} {'nDCG':>8}")
    for name in STRATEGIES:
        s = summary[name]
        print(f"{name:<10} {s[f'recall@{k}']:>8.3f} {s[f'MRR@{k}']:>8.3f} {s[f'nDCG@{k}']:>8.3f}")

    # Per-category recall (this is where multi-hop tells the story)
    print(f"\n----- recall@{k} by question type -----")
    cats = ["single_hop", "multi_hop", "temporal"]
    print(f"{'strategy':<10}" + "".join(f"{c:>12}" for c in cats))
    for name in STRATEGIES:
        line = f"{name:<10}"
        for c in cats:
            vals = per_cat[name].get(c, [])
            line += f"{mean_ignore_nan(vals):>12.3f}" if vals else f"{'-':>12}"
        print(line)

    # Save per-question CSV
    import csv
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "qid", "category", "recall", "rr", "ndcg", "retrieved"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved per-question results to: {OUT}")
    print("Use these numbers in ADR-2 (retrieval strategy decision).")


if __name__ == "__main__":
    main()
