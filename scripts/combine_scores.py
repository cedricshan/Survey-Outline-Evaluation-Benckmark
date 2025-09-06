import json
from typing import Any, Dict, List, Tuple

from ref_reward import article_reward

try:
    from zss import simple_distance, Node
except ImportError as e:
    raise SystemExit("Missing dependency 'zss'. Please install with: pip install zss")


def _build_shape_tree_from_sections(sections: List[Dict[str, Any]]) -> Node:
    """
    Build a zss.Node tree using only structure (levels) from a section list.
    Each section may contain a 'level' field; non-int or missing levels default to 1.
    """
    root = Node("root")
    stack: List[Tuple[int, Node]] = [(0, root)]
    for item in sections or []:
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


def compute_shape_and_reward(human_sections: List[Dict[str, Any]],
                             model_sections: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Compute (shape_distance, reward) for a single pair of outlines.

    Inputs:
      - human_sections: list of section dicts (true)
      - model_sections: list of section dicts (pred)

    Returns:
      (shape_distance, R_article)
    """
    # 1) Shape-only distance (tree edit distance on levels only)
    human_tree = _build_shape_tree_from_sections(human_sections)
    model_tree = _build_shape_tree_from_sections(model_sections)

    # raw edit operations count
    edit_ops = float(simple_distance(human_tree, model_tree))

    # node counts for normalization: TED = editDistance / max(|Ni|, |Nj|)
    def _count_nodes(n: Node) -> int:
        c = 1
        for ch in n.children:
            c += _count_nodes(ch)
        return c

    nodes_h = _count_nodes(human_tree)
    nodes_m = _count_nodes(model_tree)
    denom = max(nodes_h, nodes_m) if max(nodes_h, nodes_m) > 0 else 1
    shape_distance = edit_ops / float(denom)

    # 2) Reward based on refs and numbering/title mapping
    R_article, _ = article_reward(gen_sections=model_sections, true_sections=human_sections, beta=0.5, Lmax=None, quadratic=True, return_details=True)
    return shape_distance, float(R_article)


if __name__ == "__main__":
    # More complex example to validate (different structures, includes level=3)
    human = [
        {"level": 1, "numbering": "1", "title": "Intro", "ref": ["A", "B"]},
        {"level": 2, "numbering": "1.1", "title": "Background", "ref": ["C"]},
        {"level": 2, "numbering": "1.2", "title": "Method", "ref": ["D", "E"]},
        {"level": 3, "numbering": "1.2.1", "title": "Detail", "ref": ["F"]},
    ]
    model = [
        {"level": 1, "numbering": "1", "title": "Intro", "ref": ["A", "B"]},
        {"level": 2, "numbering": "1.1", "title": "Background", "ref": ["C"]},
        {"level": 2, "numbering": "1.2", "title": "Method", "ref": ["D", "E"]},
        {"level": 3, "numbering": "1.2.1", "title": "Detail", "ref": ["F"]},
    ]

    dist, reward = compute_shape_and_reward(human, model)
    print(json.dumps({"shape_distance": dist, "reward": reward}, ensure_ascii=False))


