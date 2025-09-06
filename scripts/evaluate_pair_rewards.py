#!/usr/bin/env python3
"""
Batch compute two metrics between human and model outlines for the whole dataset:
- shape_distance: Zhang–Shasha tree edit distance (structure only)
- reward: article-level reward from ref_reward.article_reward

It leverages the single-pair API from combine_scores.compute_shape_and_reward.

Usage:
  python scripts/evaluate_pair_rewards.py \
    --human_file datasets/human_generation.normalized.jsonl \
    --model_file outputs/run_20250904_155708/generation.normalized.jsonl \
    --output outputs/run_20250904_155708/human_model.rewards.jsonl \
    --max_workers 16
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, DefaultDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from combine_scores import compute_shape_and_reward


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_sections(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the sections list from a normalized item (uses 'outline')."""
    sections = item.get("outline", [])
    # ensure list of dicts
    if isinstance(sections, list):
        processed_sections = []
        for s in sections:
            if isinstance(s, dict):
                # Convert "number" field to "numbering" field for compatibility
                processed_section = s.copy()
                if "number" in processed_section and "numbering" not in processed_section:
                    processed_section["numbering"] = processed_section.pop("number")
                processed_sections.append(processed_section)
        return processed_sections
    return []


def process_pair(human_item: Dict[str, Any], model_item: Dict[str, Any]) -> Dict[str, Any]:
    human_sections = _extract_sections(human_item)
    model_sections = _extract_sections(model_item)
    shape_dist, reward = compute_shape_and_reward(human_sections, model_sections)
    return {
        "human_id": str(human_item.get("id", "unknown")),
        "model_id": str(model_item.get("id", "unknown")),
        "human_topic": human_item.get("topic", "Unknown Topic"),
        "model_topic": model_item.get("topic", "Unknown Topic"),
        "shape_distance": float(shape_dist),
        "reward": float(reward)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute shape_distance and reward for human vs model JSONL")
    parser.add_argument("--human_file", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    human_items = _read_jsonl(args.human_file)
    model_items = _read_jsonl(args.model_file)

    # Build maps: id -> item, and topic -> list (for fallback)
    from collections import defaultdict
    human_by_topic: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_by_topic: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    human_by_id: Dict[str, Dict[str, Any]] = {}
    model_by_id: Dict[str, Dict[str, Any]] = {}

    for it in human_items:
        topic = str(it.get("topic", "")).strip()
        hid = str(it.get("id", "")).strip()
        if topic:
            human_by_topic[topic].append(it)
        if hid:
            human_by_id[hid] = it
    for it in model_items:
        topic = str(it.get("topic", "")).strip()
        mid = str(it.get("id", "")).strip()
        if topic:
            model_by_topic[topic].append(it)
        if mid:
            model_by_id[mid] = it

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    # First, try to pair by id (support optional offset on model ids)
    model_index = model_by_id
    common_ids = [k for k in human_by_id.keys() if k in model_index]
    used_ids = set()
    for k in common_ids:
        pairs.append((human_by_id[k], model_index[k]))
        used_ids.add(k)

    # Fallback: pair remaining by topic and order
    common_topics = [t for t in human_by_topic.keys() if t in model_by_topic]
    for t in common_topics:
        h_list = [x for x in human_by_topic[t] if str(x.get("id", "")) not in used_ids]
        m_list = [x for x in model_by_topic[t] if str(x.get("id", "")) not in used_ids]
        for i in range(min(len(h_list), len(m_list))):
            pairs.append((h_list[i], m_list[i]))

    total = len(pairs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results: List[Dict[str, Any]] = []
    with open(args.output, 'w', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_pair, h, m): idx for idx, (h, m) in enumerate(pairs)}
            with tqdm(total=total, desc="Computing rewards") as pbar:
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    pbar.update(1)

    # summary
    if total:
        avg_shape = sum(r["shape_distance"] for r in results) / total
        avg_reference_accuracy = sum(r["reward"] for r in results) / total
    else:
        avg_shape = 0.0
        avg_reference_accuracy = 0.0
    summary_path = args.output.replace('.jsonl', '.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({"pairs": total, "avg_shape_distance": avg_shape, "avg_reference_accuracy": avg_reference_accuracy}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"pairs": total, "avg_shape_distance": avg_shape, "avg_reference_accuracy": avg_reference_accuracy}, ensure_ascii=False))


if __name__ == "__main__":
    main()


