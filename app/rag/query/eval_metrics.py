from typing import Any


def _normalize_chunk_ids(chunk_ids: list[Any] | None) -> list[str]:
    if not chunk_ids:
        return []
    return [str(chunk_id) for chunk_id in chunk_ids if chunk_id is not None]


def _deduplicate_keep_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_chunk_ids(docs: list[dict] | None) -> list[str]:
    if not docs:
        return []
    chunk_ids: list[str] = []
    for doc in docs:
        chunk_id = doc.get("chunk_id")
        if chunk_id is None:
            continue
        chunk_ids.append(str(chunk_id))
    return _deduplicate_keep_order(chunk_ids)


def compute_item_name_hit_rate(predicted_item_names: list[str] | None, expected_item_names: list[str] | None) -> float:
    predicted = set(predicted_item_names or [])
    expected = set(expected_item_names or [])
    if not expected:
        return 0.0
    return len(predicted & expected) / len(expected)


def compute_chunk_metrics(
    retrieved_chunk_ids: list[Any] | None,
    gold_chunk_ids: list[Any] | None,
    must_hit_chunk_ids: list[Any] | None = None,
) -> dict[str, Any]:
    retrieved = _deduplicate_keep_order(_normalize_chunk_ids(retrieved_chunk_ids))
    gold = _deduplicate_keep_order(_normalize_chunk_ids(gold_chunk_ids))
    must_hit = _deduplicate_keep_order(_normalize_chunk_ids(must_hit_chunk_ids))

    gold_set = set(gold)
    must_hit_set = set(must_hit)
    hit_chunk_ids = [chunk_id for chunk_id in retrieved if chunk_id in gold_set]
    must_hit_ids = [chunk_id for chunk_id in retrieved if chunk_id in must_hit_set]

    precision = len(hit_chunk_ids) / len(retrieved) if retrieved else 0.0
    recall = len(hit_chunk_ids) / len(gold_set) if gold_set else 0.0
    must_hit_rate = len(must_hit_ids) / len(must_hit_set) if must_hit_set else 0.0

    return {
        "retrieved_chunk_ids": retrieved,
        "retrieved_count": len(retrieved),
        "gold_chunk_ids": gold,
        "gold_count": len(gold),
        "hit_chunk_ids": hit_chunk_ids,
        "hit_count": len(hit_chunk_ids),
        "must_hit_chunk_ids": must_hit,
        "must_hit_ids": must_hit_ids,
        "must_hit_count": len(must_hit_ids),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "must_hit_rate": round(must_hit_rate, 4),
    }


def evaluate_query_state(result_state: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_item_names = expected.get("expected_item_names", [])
    gold_chunk_ids = expected.get("gold_chunk_ids", [])
    must_hit_chunk_ids = expected.get("must_hit_chunk_ids", [])

    layers = {}
    for layer_name in ("embedding_chunks", "hyde_embedding_chunks", "rrf_chunks", "reranked_docs"):
        layers[layer_name] = compute_chunk_metrics(
            retrieved_chunk_ids=extract_chunk_ids(result_state.get(layer_name, [])),
            gold_chunk_ids=gold_chunk_ids,
            must_hit_chunk_ids=must_hit_chunk_ids,
        )

    return {
        "case_id": expected.get("case_id", ""),
        "question": expected.get("question", ""),
        "expected_item_names": expected_item_names,
        "predicted_item_names": result_state.get("item_names", []),
        "item_name_hit_rate": round(
            compute_item_name_hit_rate(result_state.get("item_names", []), expected_item_names),
            4,
        ),
        "layers": layers,
    }