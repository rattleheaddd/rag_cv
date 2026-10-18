def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not relevant_ids:
        raise ValueError("no relevant ids were found")

    relevant_ids = set(relevant_ids)
    retrieved_ids = set(retrieved_ids[:k])
    overlap = retrieved_ids & relevant_ids

    return len(overlap) / len(relevant_ids)

def reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not relevant_ids:
        raise ValueError("no relevant ids were found")
    
    relevant_ids = set(relevant_ids)

    for ind, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1 / ind

    return 0

def evaluate(test_cases: list[dict], k: int = 3) -> dict:
    if not test_cases:
        raise ValueError("test_cases empty")

    n = len(test_cases)
    recalls = 0
    rrs = 0

    for case in test_cases:
        recalls += (recall_at_k(case["retrieved_ids"], case["relevant_ids"], k))
        rrs += (reciprocal_rank(case["retrieved_ids"], case["relevant_ids"], k))
        
    return {f"mean_recall_at_{k}": recalls / n, f"mrr@{k}": rrs / n}


if __name__ == "__main__":
    test_cases = [
        {
            "retrieved_ids": [2, 1, 0],
            "relevant_ids": [1, 2]
        },
        {
            "retrieved_ids": [0, 2, 1],
            "relevant_ids": [1, 2]
        },
        {
            "retrieved_ids": [0, 3, 4],
            "relevant_ids": [1, 2]
        }
    ]

    print(evaluate(test_cases, k=3))