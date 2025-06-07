"""
ML Test Score, based on Google's methodology:
- Feature and Data Integrity tests
- Model Development tests
- ML Infrastructure tests
- Monitoring tests
"""

import json
from pathlib import Path

CATEGORY_FILES = {
    "data": ["tests/data/test_data.py"],
    "model": [
        "tests/development/test_model.py",
        "tests/development/test_metamorphic.py",
    ],
    "infrastructure": ["tests/infrastructure/test_infrastructure.py"],
    "monitor": ["tests/monitoring/test_monitoring.py"],
}

CATEGORY_CONFIG = {
    "data": {
        "weight": 30,
        "description": "Feature and Data Integrity",
        "keywords": [
            "data", "feature", "features", "schema",
            "nulls", "class", "length", "empty",
            "duplicate", "invalid", "columns",
            "variance", "dataset", "distributions",
            "relationship", "privacy"
        ],
    },
    "model": {
        "weight": 35,
        "description": "Model Development",
        "keywords": [
            "model", "negation", "nondeterminism",
            "robustness", "slice", "common",
            "typos", "irrelevance", "named",
            "fairness", "gendered", "short", "long",
            "whitespace", "punctuation", "synonym",
            "monotonicity", "permutation", "generalization",
            "repair", "hyperparameter","bias"
        ],
    },
    "infrastructure": {
        "weight": 20,
        "description": "ML Infrastructure",
        "keywords": [
            "artifacts", "consistency", "serving",
            "quality", "latency", "memory", "staleness",
            "inference", "extraction"
            "limit", "pipeline", "previous", "serving"
        ],
    },
    "monitor": {
        "weight": 15,
        "description": "Monitoring",
        "keywords": [
            "monitor", "invariance", "staleness", "perturbation",
            "inference", "latency",
        ],
    },
}


def check_keywords_in_files(keywords, file_paths):
    """Return the set of matched keywords found in the given files."""
    found_keywords = set()

    for path in file_paths:
        file = Path(path)
        if not file.exists():
            continue
        content = file.read_text(encoding="utf-8").lower()

        for kw in keywords:
            if kw.lower() in content:
                found_keywords.add(kw)

    return found_keywords

def calculate_adequacy():
    """Compute per-category and overall ML test adequacy based on keyword presence."""
    scores = {}
    total_weighted_score = 0
    total_weight = 0

    for category, config in CATEGORY_CONFIG.items():
        files = CATEGORY_FILES.get(category, [])
        keywords = config["keywords"]
        found = check_keywords_in_files(keywords, files)

        total = len(keywords)
        found_count = len(found)
        score = (found_count / total) * 100 if total > 0 else 0.0

        scores[category] = {
            "score": round(score, 1),
            "found_keywords": sorted(found),
            "missed_keywords": sorted(set(keywords) - found),
            "description": config["description"],
            "weight": config["weight"],
            "files": files,
        }

        total_weighted_score += score * config["weight"]
        total_weight += config["weight"]

    overall_score = round(total_weighted_score / total_weight, 1)
    return overall_score, scores


def main():
    """Main entry point for computing ML test adequacy."""

    overall, details = calculate_adequacy()

    print(f"\n\n🔎 Test adequacy: ML Test Score= {overall}/100\n")
    for _, info in details.items():
        found = len(info["found_keywords"])
        total = found + len(info["missed_keywords"])
        print(
            f"{info['description']:<25}: {info['score']:>5}% "
            f"({found}/{total} keywords found)"
        )

    with open("metrics/ml_test_score.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_score": overall,
            "categories": details
        }, f, indent=2)

    print("\n📁 Saved keyword coverage report to: metrics/ml_keyword_score.json")


if __name__ == "__main__":
    main()
