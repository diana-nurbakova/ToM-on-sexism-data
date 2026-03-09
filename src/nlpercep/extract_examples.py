"""Extract qualitative examples for the paper.

Three example types:
1. Detection–interpretation dissociation (tweets)
2. Gender-differential categorization (tweets)
3. Meme objectification gender split (memes)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nlpercep.data import load_dataset, load_memes_dataset

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

HARM_CATS = {"SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"}
COGNITIVE_CATS = {"STEREOTYPING-DOMINANCE", "IDEOLOGICAL-INEQUALITY"}


def _get_cats(row, i: int) -> set[str]:
    """Get category set for annotator i (only if they said YES)."""
    if row[f"ann_{i}_task1_1"] != "YES":
        return set()
    cats = row[f"ann_{i}_task1_3"]
    if isinstance(cats, list):
        return {c for c in cats if c != "-"}
    return set() if cats == "-" else {cats}


def _annotator_summary(row, i: int) -> dict:
    """Build a summary dict for one annotator."""
    t13 = row[f"ann_{i}_task1_3"]
    if isinstance(t13, list):
        t13 = t13.copy()
    return {
        "annotator_index": i,
        "gender": row[f"ann_{i}_gender"],
        "task1_1": row[f"ann_{i}_task1_1"],
        "task1_2": row[f"ann_{i}_task1_2"],
        "task1_3": t13,
    }


def _build_example(row, example_type: str, extra: dict | None = None) -> dict:
    """Build a standard example dict."""
    ex = {
        "id": row["id"],
        "lang": row["lang"],
        "text": row["text"],
        "yes_count": int(row["yes_count"]),
        "example_type": example_type,
        "annotators": [_annotator_summary(row, i) for i in range(6)],
    }
    if extra:
        ex.update(extra)
    return ex


# ── Type 1: Detection–interpretation dissociation ────────────────────────────

def extract_type1(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Strong YES agreement + intent disagreement + category divergence."""
    candidates = df[df["yes_count"] >= 5].copy()

    results = []
    for _, row in candidates.iterrows():
        # Collect intent labels among YES annotators
        intents = set()
        all_cats = set()
        for i in range(6):
            if row[f"ann_{i}_task1_1"] == "YES":
                intents.add(row[f"ann_{i}_task1_2"])
                cats = row[f"ann_{i}_task1_3"]
                if isinstance(cats, list):
                    all_cats.update(c for c in cats if c != "-")

        if len(intents) < 2 or len(all_cats) < 3:
            continue

        results.append((row, len(intents), len(all_cats)))

    # Sort: prefer English, then most diverse intents, then most diverse cats
    results.sort(key=lambda x: (x[0]["lang"] == "en", x[1], x[2]), reverse=True)

    examples = []
    for row, n_intents, n_cats in results[:n]:
        examples.append(_build_example(row, "detection_interpretation_dissociation", {
            "n_distinct_intents": n_intents,
            "n_distinct_categories": n_cats,
            "entropy_1_2": round(float(row["entropy_1_2"]), 4) if pd.notna(row["entropy_1_2"]) else None,
            "jaccard_1_3": round(float(row["jaccard_1_3"]), 4) if pd.notna(row["jaccard_1_3"]) else None,
        }))
    return examples


# ── Type 2: Gender-differential categorization ───────────────────────────────

def extract_type2(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Both genders agree sexist, but F→harm cats, M→cognitive cats (or vice versa)."""
    candidates = df[df["yes_count"] >= 5].copy()

    results = []
    for _, row in candidates.iterrows():
        f_cats = set()
        m_cats = set()
        for i in range(3):
            f_cats |= _get_cats(row, i)
        for i in range(3, 6):
            m_cats |= _get_cats(row, i)

        if not f_cats or not m_cats:
            continue

        # Score: F has harm that M doesn't + M has cognitive that F doesn't
        # (or the reverse pattern)
        pattern_a = len((f_cats & HARM_CATS) - m_cats) + len((m_cats & COGNITIVE_CATS) - f_cats)
        pattern_b = len((m_cats & HARM_CATS) - f_cats) + len((f_cats & COGNITIVE_CATS) - m_cats)
        score = max(pattern_a, pattern_b)

        if score == 0:
            continue

        direction = "F_harm_M_cognitive" if pattern_a >= pattern_b else "M_harm_F_cognitive"
        results.append((row, score, direction, f_cats, m_cats))

    results.sort(key=lambda x: (x[0]["lang"] == "en", x[1]), reverse=True)

    examples = []
    for row, score, direction, f_cats, m_cats in results[:n]:
        examples.append(_build_example(row, "gender_differential_categorization", {
            "divergence_score": score,
            "direction": direction,
            "f_categories": sorted(f_cats),
            "m_categories": sorted(m_cats),
        }))
    return examples


# ── Type 3: Meme objectification gender split ────────────────────────────────

def extract_type3(memes_df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Majority YES memes where F assigns OBJECTIFICATION more than M."""
    candidates = memes_df[memes_df["yes_count"] >= 4].copy()

    results = []
    for _, row in candidates.iterrows():
        f_obj = 0
        f_yes = 0
        m_obj = 0
        m_yes = 0
        for i in range(3):
            if row[f"ann_{i}_task1_1"] == "YES":
                f_yes += 1
                cats = _get_cats(row, i)
                if "OBJECTIFICATION" in cats:
                    f_obj += 1
        for i in range(3, 6):
            if row[f"ann_{i}_task1_1"] == "YES":
                m_yes += 1
                cats = _get_cats(row, i)
                if "OBJECTIFICATION" in cats:
                    m_obj += 1

        if f_yes == 0:
            continue

        f_rate = f_obj / f_yes
        m_rate = m_obj / m_yes if m_yes > 0 else 0.0

        # F assigns OBJECTIFICATION but M doesn't (or at lower rate)
        if f_obj > 0 and f_rate > m_rate:
            gap = f_rate - m_rate
            results.append((row, gap, f_rate, m_rate, f_obj, m_obj))

    results.sort(key=lambda x: (x[0]["lang"] == "en", x[1]), reverse=True)

    examples = []
    for row, gap, f_rate, m_rate, f_obj, m_obj in results[:n]:
        examples.append(_build_example(row, "meme_objectification_gender_split", {
            "f_objectification_count": f_obj,
            "m_objectification_count": m_obj,
            "f_objectification_rate": round(f_rate, 3),
            "m_objectification_rate": round(m_rate, 3),
            "rate_gap": round(gap, 3),
        }))
    return examples


def main():
    print("Loading tweets dataset...")
    df = load_dataset()
    print(f"  {len(df)} instances loaded.")

    print("Loading memes dataset...")
    memes_df = load_memes_dataset()
    print(f"  {len(memes_df)} instances loaded.")

    print("\nExtracting Type 1: Detection–interpretation dissociation...")
    type1 = extract_type1(df)
    print(f"  Found {len(type1)} examples.")

    print("Extracting Type 2: Gender-differential categorization...")
    type2 = extract_type2(df)
    print(f"  Found {len(type2)} examples.")

    print("Extracting Type 3: Meme objectification gender split...")
    type3 = extract_type3(memes_df)
    print(f"  Found {len(type3)} examples.")

    output = {
        "description": "Qualitative examples for the NLPercep paper",
        "type1_detection_interpretation_dissociation": {
            "description": "Strong detection agreement (≥5/6 YES) with intent disagreement (≥2 labels) and category divergence (≥3 categories). English preferred.",
            "count": len(type1),
            "examples": type1,
        },
        "type2_gender_differential_categorization": {
            "description": "Both genders agree sexist (≥5/6 YES) but F annotators assign harm categories while M assign cognitive categories (or vice versa). English preferred.",
            "count": len(type2),
            "examples": type2,
        },
        "type3_meme_objectification_gender_split": {
            "description": "Majority-YES memes where F annotators assign OBJECTIFICATION at higher rates than M annotators. English preferred.",
            "count": len(type3),
            "examples": type3,
        },
    }

    out_path = OUTPUT_DIR / "qualitative_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    main()
