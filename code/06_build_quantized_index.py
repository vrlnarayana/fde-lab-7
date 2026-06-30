"""
06_build_quantized_index.py  (LAB 7 — additional step / assignment)
-------------------------------------------------------------------
Builds an Azure AI Search index with VECTOR QUANTIZATION (compression) and
measures the trade-off: storage saved vs retrieval quality kept.

It is self-contained (uses the azure-search-documents SDK directly, not
LangChain) so you can see exactly how quantization is configured, and so the
evaluation does not depend on any LangChain field naming.

Three modes (run each, then compare):
    --kind none     baseline: full-precision float32 vectors, no compression
    --kind scalar   scalar quantization (int8, ~4x smaller)
    --kind binary   binary quantization (1 bit, up to ~28x smaller)

What it does each run:
    1. (Re)creates an index with the chosen compression on the vector field.
    2. Embeds the corpus chunks with Azure OpenAI and uploads them.
    3. Runs the 50-question golden set with a vector query (oversampling on).
    4. Prints recall@k / MRR@k / nDCG@k AND the index storage size.

Examples:
    python 06_build_quantized_index.py --kind none   --index cp-none   --eval
    python 06_build_quantized_index.py --kind scalar --index cp-scalar --eval --oversampling 10
    python 06_build_quantized_index.py --kind binary --index cp-binary --eval --oversampling 10

Notes:
  * Uses the STABLE azure-search-documents 11.5.2 API. In this version the
    rescoring knobs are 'rerank_with_original_vectors' + 'default_oversampling'.
    (The newer 'rescoringOptions / discardOriginals' form is preview-only and
    is explained in the assignment document.)
  * HNSW is required for rescoring/oversampling, so we use HNSW.
  * text-embedding-3-large returns 3072-dim vectors -> a great fit for binary
    quantization (recommended for dimensions > 1024).
"""
import argparse, json, os, time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, HnswParameters,
    VectorSearchAlgorithmMetric, ScalarQuantizationCompression,
    BinaryQuantizationCompression, ScalarQuantizationParameters,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

from corpus_loader import build_chunk_records
from azure_clients import get_embeddings
from eval_metrics import recall_at_k, reciprocal_rank_at_k, ndcg_at_k, mean_ignore_nan

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.abspath(os.path.join(HERE, "..", "golden_set", "golden_set.json"))
EMBED_DIM = 3072  # text-embedding-3-large

PROFILE = "vprofile"
ALGO = "hnsw-algo"
COMP_SCALAR = "sq-config"
COMP_BINARY = "bq-config"
VECTOR_FIELD = "content_vector"


def endpoint_and_cred():
    ep = os.environ["AZURE_SEARCH_ENDPOINT"]
    cred = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    return ep, cred


def build_vector_search(kind: str, oversampling: float) -> VectorSearch:
    """Create the VectorSearch block, wiring compression onto the profile."""
    algorithms = [HnswAlgorithmConfiguration(
        name=ALGO,
        parameters=HnswParameters(m=4, ef_construction=400, ef_search=500,
                                  metric=VectorSearchAlgorithmMetric.COSINE),
    )]

    compressions, compression_name = [], None
    if kind == "scalar":
        compression_name = COMP_SCALAR
        compressions = [ScalarQuantizationCompression(
            compression_name=COMP_SCALAR,
            rerank_with_original_vectors=True,      # rescore using full-precision vectors
            default_oversampling=oversampling,      # pull extra candidates, then rescore
            parameters=ScalarQuantizationParameters(quantized_data_type="int8"),
        )]
    elif kind == "binary":
        compression_name = COMP_BINARY
        compressions = [BinaryQuantizationCompression(
            compression_name=COMP_BINARY,
            rerank_with_original_vectors=True,
            default_oversampling=oversampling,
        )]
    # kind == "none" -> no compression, full precision

    profile = VectorSearchProfile(
        name=PROFILE,
        algorithm_configuration_name=ALGO,
        compression_name=compression_name,          # None for baseline
    )
    return VectorSearch(algorithms=algorithms, profiles=[profile], compressions=compressions)


def make_index(index_name: str, kind: str, oversampling: float) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="pathway_code", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="year", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="status", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="section", type=SearchFieldDataType.String),
        SearchableField(name="specialty", type=SearchFieldDataType.String),
        SimpleField(name="citation", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name=PROFILE,
        ),
    ]
    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=build_vector_search(kind, oversampling),
    )


def recreate_index(index_name: str, kind: str, oversampling: float):
    ep, cred = endpoint_and_cred()
    sic = SearchIndexClient(ep, cred)
    try:
        sic.delete_index(index_name)
        print(f"Deleted existing index '{index_name}'.")
    except Exception:
        pass
    sic.create_index(make_index(index_name, kind, oversampling))
    print(f"Created index '{index_name}' (kind={kind}).")


def upload_chunks(index_name: str):
    ep, cred = endpoint_and_cred()
    sc = SearchClient(ep, cred, index_name)
    emb = get_embeddings()

    chunks = build_chunk_records()
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with Azure OpenAI...")
    vectors = emb.embed_documents(texts)

    docs = []
    for c, v in zip(chunks, vectors):
        docs.append({
            "id": c["chunk_id"], "parent_id": c["parent_id"],
            "pathway_code": c["pathway_code"], "version": c["version"],
            "year": int(c["year"]), "status": c["status"],
            "title": c["title"], "section": c["section"],
            "specialty": c["specialty"], "citation": c["citation"],
            "content": c["content"], VECTOR_FIELD: v,
        })
    # upload in batches
    B = 100
    for i in range(0, len(docs), B):
        sc.upload_documents(documents=docs[i:i + B])
    print(f"Uploaded {len(docs)} chunks to '{index_name}'.")


def index_storage(index_name: str):
    ep, cred = endpoint_and_cred()
    sic = SearchIndexClient(ep, cred)
    stats = sic.get_index_statistics(index_name)
    # keys: document_count, storage_size, vector_index_size (bytes)
    return stats


def evaluate(index_name: str, k: int, oversampling: float):
    ep, cred = endpoint_and_cred()
    sc = SearchClient(ep, cred, index_name)
    emb = get_embeddings()
    with open(GOLD, encoding="utf-8") as f:
        gold = [q for q in json.load(f) if q["category"] != "negative"]

    rec, mrr, ndcg = [], [], []
    for q in gold:
        qv = emb.embed_query(q["question"])
        vq = VectorizedQuery(vector=qv, k_nearest_neighbors=k + 5,
                             fields=VECTOR_FIELD, oversampling=oversampling)
        results = sc.search(search_text=None, vector_queries=[vq],
                            select=["parent_id"], top=k + 5)
        seen, ordered = set(), []
        for r in results:
            pid = r["parent_id"]
            if pid not in seen:
                seen.add(pid); ordered.append(pid)
        got = ordered[:k]
        rel = q["relevant_doc_ids"]
        rec.append(recall_at_k(got, rel, k))
        mrr.append(reciprocal_rank_at_k(got, rel, k))
        ndcg.append(ndcg_at_k(got, rel, k))
    return mean_ignore_nan(rec), mean_ignore_nan(mrr), mean_ignore_nan(ndcg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["none", "scalar", "binary"], required=True)
    ap.add_argument("--index", required=True, help="index name to create/use")
    ap.add_argument("--oversampling", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="run golden-set evaluation")
    ap.add_argument("--skip-build", action="store_true", help="evaluate an existing index only")
    args = ap.parse_args()

    if not args.skip_build:
        recreate_index(args.index, args.kind, args.oversampling)
        upload_chunks(args.index)
        print("Waiting 5s for indexing to settle...")
        time.sleep(5)

    stats = index_storage(args.index)
    storage_mb = (stats.get("storage_size") or 0) / (1024 * 1024)
    vector_mb = (stats.get("vector_index_size") or 0) / (1024 * 1024)
    print(f"\nINDEX '{args.index}' (kind={args.kind})")
    print(f"  documents      : {stats.get('document_count')}")
    print(f"  storage_size   : {storage_mb:.3f} MB")
    print(f"  vector_index   : {vector_mb:.3f} MB")

    if args.eval:
        r, m, n = evaluate(args.index, args.k, args.oversampling)
        print(f"  recall@{args.k}      : {r:.3f}")
        print(f"  MRR@{args.k}         : {m:.3f}")
        print(f"  nDCG@{args.k}        : {n:.3f}")

    print("\nTip: run kind=none, scalar and binary, then fill the comparison")
    print("table in the assignment (storage saved vs recall kept).")


if __name__ == "__main__":
    main()
