"""E5 commitment/consistency exemplar pools + fixed pairing schedule (§3.3, §7.3).

Two pools per language, drawn from unanimous (6/0) EXIST items — clearly-sexist and
clearly-not-sexist — disjoint from the E5 target items (and any excluded ids, so the
exemplars never overlap E1/E2/E5 targets, §7.3). Each target item is paired with one
exemplar per direction, rotated deterministically by item index so no single
exemplar over-anchors the effect (§7.3); the schedule is fixed in advance and stored
alongside the items.
"""

from __future__ import annotations

import pandas as pd

from ..data import load_pool

N_PER_POOL = 5
SEED = 42


def select_pools(df: pd.DataFrame, lang: str, exclude_ids: set | None = None,
                 n_per: int = N_PER_POOL, seed: int = SEED) -> dict[str, list[dict]]:
    """Return ``{"sexist": [...], "not_sexist": [...]}`` of unanimous exemplars.

    Each entry is ``{"item_id": str, "text": str}``. "Unanimous" means a 6/0
    detection split: every annotator YES (clearly sexist) or every annotator NO
    (clearly not sexist), read off the gold soft label.
    """
    exclude_ids = {str(i) for i in (exclude_ids or set())}
    pool = df[(df["lang"] == lang) & (~df["item_id"].isin(exclude_ids))].copy()
    pool["p_sexist"] = pool["gold_1_1"].apply(lambda d: d["sexist"])
    out: dict[str, list[dict]] = {}
    for direction, mask in (("sexist", pool["p_sexist"] >= 1.0),
                            ("not_sexist", pool["p_sexist"] <= 0.0)):
        cell = pool[mask]
        take = min(n_per, len(cell))
        chosen = cell.sample(n=take, random_state=seed) if take > 0 else cell
        out[direction] = [{"item_id": str(r["item_id"]), "text": r["text"]}
                          for _, r in chosen.iterrows()]
    return out


def build_pairing(item_ids, pools: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
    """Fixed exemplar pairing: ``{item_id: {"sexist": text, "not_sexist": text}}``.

    Rotates through each pool by the target item's position so the pairing is
    deterministic and spreads exemplars evenly across the targets (§7.3).
    """
    sexist = [e["text"] for e in pools["sexist"]]
    not_sexist = [e["text"] for e in pools["not_sexist"]]
    if not sexist or not_sexist == []:
        raise ValueError("both exemplar pools must be non-empty for commitment cells")
    pairing: dict[str, dict[str, str]] = {}
    for i, iid in enumerate(item_ids):
        pairing[str(iid)] = {
            "sexist": sexist[i % len(sexist)],
            "not_sexist": not_sexist[i % len(not_sexist)],
        }
    return pairing


def build_pairing_for_sample(item_ids, df: pd.DataFrame | None = None, lang: str = "en",
                             exclude_ids: set | None = None, seed: int = SEED
                             ) -> tuple[dict[str, dict[str, str]], dict[str, list[dict]]]:
    """Convenience: select pools (excluding the target ids) and build the pairing.

    Returns ``(pairing, pools)``. The target ids are themselves excluded from the
    exemplar pools so an item is never its own precedent.
    """
    if df is None:
        df = load_pool(lang)
    excl = {str(i) for i in (exclude_ids or set())} | {str(i) for i in item_ids}
    pools = select_pools(df, lang, exclude_ids=excl, seed=seed)
    return build_pairing(item_ids, pools), pools
