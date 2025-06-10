# co_ai/utils/graph_tools.py

def build_mermaid_graph(trace, title="Reasoning Graph"):
    """
    Given a list of steps (or a dict-based graph), return a Mermaid flowchart.
    """
    lines = ["graph TD"]
    for i in range(len(trace) - 1):
        from_node = f"Step{i}[{trace[i]}]"
        to_node = f"Step{i+1}[{trace[i+1]}]"
        lines.append(f"{from_node} --> {to_node}")
    return "\n".join(lines)


def compare_graphs(graph1, graph2):
    """
    Compares two lists of nodes (or strings) and returns:
    - matched nodes
    - only in graph1
    - only in graph2
    """
    set1 = set(graph1)
    set2 = set(graph2)

    matches = list(set1 & set2)
    only_1 = list(set1 - set2)
    only_2 = list(set2 - set1)

    return matches, only_1, only_2


def analyze_graph_impact(graph1, graph2, score_lookup_fn):
    """
    Returns a list of dictionaries summarizing overlap and score delta.
    """
    matches, only_1, only_2 = compare_graphs(graph1, graph2)
    results = []

    for node in matches:
        score1 = score_lookup_fn(node, source="graph1")
        score2 = score_lookup_fn(node, source="graph2")
        results.append({
            "node": node,
            "type": "match",
            "delta": score2 - score1
        })

    for node in only_1:
        results.append({
            "node": node,
            "type": "only_graph1",
            "score": score_lookup_fn(node, source="graph1")
        })

    for node in only_2:
        results.append({
            "node": node,
            "type": "only_graph2",
            "score": score_lookup_fn(node, source="graph2")
        })

    return results
