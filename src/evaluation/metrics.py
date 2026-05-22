
def hit_rate_at_k(actual_items, recommended_items, k=10):
    actual_items = set(actual_items)
    recommended_items = set(recommended_items[:k])

    if len(actual_items) == 0:
        return 0.0

    hits = len(actual_items.intersection(recommended_items))
    return hits / len(actual_items)