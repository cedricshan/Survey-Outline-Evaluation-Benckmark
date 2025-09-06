#!/usr/bin/env python3
"""
Compare outline shapes (tree structure only) between human and model outlines.

- Shape-only distance uses Zhang–Shasha tree edit distance with labels removed
  (all nodes treated as identical). This evaluates structural similarity only.

Features:
- process_item(human_item: dict, model_item: dict) -> dict
- Batch evaluation over two JSONL files
- CLI with progress and output saving
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from zss import simple_distance, Node
except ImportError as e:
    raise SystemExit("Missing dependency 'zss'. Please install with: pip install zss")


def _build_shape_tree(obj: Dict[str, Any]) -> Node:
    """Build a zss.Node tree using only structure (labels omitted)."""
    root = Node("root")
    outline: List[Dict[str, Any]] = obj.get("outline", [])
    stack: List[Tuple[int, Node]] = [(0, root)]
    for item in outline:
        try:
            level = int(item.get("level", 1))
        except Exception:
            level = 1
        node = Node("n")
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else root
        parent.addkid(node)
        stack.append((level, node))
    return root


def _shape_cost(a: Node, b: Node) -> int:
    """Cost function: all node relabel costs are zero (shape-only)."""
    return 0


def process_item(human_item: Dict[str, Any], model_item: Dict[str, Any]) -> Dict[str, Any]:
    """Compute shape-only distance for a pair of outline items.

    Returns a dict with ids, topics, node counts and distance.
    """
    human_tree = _build_shape_tree(human_item)
    model_tree = _build_shape_tree(model_item)

    # simple_distance ignores labels for equal strings; use constant labels so relabel cost=0
    dist = simple_distance(human_tree, model_tree)

    def _count(n: Node) -> int:
        c = 1
        for ch in n.children:
            c += _count(ch)
        return c

    human_nodes = _count(human_tree)
    model_nodes = _count(model_tree)

    return {
        "human_id": str(human_item.get("id", "unknown")),
        "model_id": str(model_item.get("id", "unknown")),
        "human_topic": human_item.get("topic", "Unknown Topic"),
        "model_topic": model_item.get("topic", "Unknown Topic"),
        "human_nodes": human_nodes,
        "model_nodes": model_nodes,
        "shape_distance": float(dist)
    }


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_files(human_file: str, model_file: str, output_file: str, max_workers: int = 8) -> Dict[str, Any]:
    human_items = _read_jsonl(human_file)
    model_items = _read_jsonl(model_file)

    total = min(len(human_items), len(model_items))
    results: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_item, human_items[i], model_items[i]): i
                for i in range(total)
            }
            with tqdm(total=total, desc="Comparing shapes") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    pbar.update(1)

    # summary
    avg = sum(r["shape_distance"] for r in results) / total if total else 0.0
    summary = {"pairs": total, "avg_shape_distance": avg}
    with open(output_file.replace('.jsonl', '.summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare human vs model outline shapes (JSONL)")
    parser.add_argument("--human_file", required=True, help="Path to human JSONL (e.g., datasets/human_generation.normalized.jsonl)")
    parser.add_argument("--model_file", required=True, help="Path to model JSONL (e.g., outputs/run_xxx/generation.normalized.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL path for pair distances")
    parser.add_argument("--max_workers", type=int, default=8, help="Concurrent workers")
    args = parser.parse_args()

    summary = compare_files(args.human_file, args.model_file, args.output, args.max_workers)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()


