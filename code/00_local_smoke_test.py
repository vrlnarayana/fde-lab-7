"""
00_local_smoke_test.py
-----------------------
Runs the ENTIRE evaluation harness OFFLINE, with no Azure account and no
internet. It uses a tiny keyword-overlap retriever instead of real embeddings.

Why this exists:
  - Lets every learner confirm the corpus + golden set + metrics all work
    BEFORE spending money on Azure.
  - Gives instructors a guaranteed-runnable demo if the Wi-Fi or Azure is down.

It is deliberately a WEAK retriever (no semantics), so the scores will be
modest. That is the point: it sets a baseline that real hybrid retrieval on
Azure should beat in Lab 7-C.

Run:
    python 00_local_smoke_test.py
"""
import json, os, re
from collections import Counter
from corpus_loader import build_chunk_records
from eval_metrics import (recall_at_k, reciprocal_rank_at_k, ndcg_at_k, mean_ignore_nan)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.abspath(os.path.join(HERE, "..", "golden_set", "golden_set.json"))

K = 5
STOP = set("a an the of to in for and or with is are be on at by as from this that "
           "which when how what who whom does do should must may within above below".split())


def tokenise(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def build_index(chunks):
    idx = []
    for c in chunks:
        idx.append((c, Counter(tokenise(c["content"]))))
    return idx


def retrieve(query, index, k=K):
    qt = Counter(tokenise(query))
    scored = []
    for c, tf in index:
        score = sum(qt[t] * tf.get(t, 0) for t in qt)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    # collapse chunk hits to ordered unique parent doc ids
    seen, ordered = set(), []
    for _, c in scored:
        if c["parent_id"] not in seen:
            seen.add(c["parent_id"])
            ordered.append(c["parent_id"])
        if len(ordered) >= k:
            break
    return ordered


def main():
    chunks = build_chunk_records()
    index = build_index(chunks)
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)

    rec, mrr, ndcg = [], [], []
    answerable = [q for q in gold if q["category"] != "negative"]
    for q in answerable:
        got = retrieve(q["question"], index, K)
        rel = q["relevant_doc_ids"]
        rec.append(recall_at_k(got, rel, K))
        mrr.append(reciprocal_rank_at_k(got, rel, K))
        ndcg.append(ndcg_at_k(got, rel, K))

    print(f"Corpus: {len(chunks)} chunks | Golden set: {len(gold)} questions "
          f"({len(answerable)} answerable)")
    print(f"--- OFFLINE keyword baseline (k={K}) ---")
    print(f"recall@{K}: {mean_ignore_nan(rec):.3f}")
    print(f"MRR@{K}   : {mean_ignore_nan(mrr):.3f}")
    print(f"nDCG@{K}  : {mean_ignore_nan(ndcg):.3f}")
    print("\nThis is the floor. Hybrid + semantic on Azure should beat it.")


if __name__ == "__main__":
    main()
