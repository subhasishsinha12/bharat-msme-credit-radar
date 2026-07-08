"""
Bharat MSME Credit Radar - Graph Contagion Overlay
=====================================================
Implements the "Layer 4" graph model described in the design brief: a
borrower-buyer-supplier contagion overlay that propagates stress from a
struggling anchor buyer / cluster through the MSME units that sell into it
(e.g. a Surat embroidery cluster hit by a buyer default), independent of
each borrower's own individually-observed conduct.

Prototype scope: every synthetic borrower belongs to a `cluster_id`
(sector-geography cell, e.g. "Textile-Surat") and, within it, an
`anchor_buyer_id` sub-group (src/data_generator.py). We build a NetworkX
star graph per anchor-buyer group (hub = anchor buyer, spokes = borrowers
selling into it), compute a cluster stress index from the *already-scored*
population, and apply a small, capped PD overlay to members of clusters
whose stress index is elevated relative to the portfolio — the mechanism
by which the Radar can flag a borrower whose own numbers still look fine
but who sits in a cluster that is already slipping together.

This is a lightweight, explainable stand-in for a production graph neural
network / graph database contagion model (see README roadmap) — it uses
observable co-movement in already-scored PDs rather than a trained GNN,
but implements the same design intent: cluster-level early warning that
individual-account models cannot see.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

MIN_CLUSTER_SIZE_FOR_ALERT = 4
CONTAGION_MAX_UPLIFT = 0.15  # cap: contagion can raise a borrower's PD by at most +15% relative
ELEVATED_STRESS_MULTIPLE = 1.6  # cluster average PD vs portfolio average PD, above which a cluster is "elevated"


def build_contagion_graph(df: pd.DataFrame) -> nx.Graph:
    """Star graph per anchor_buyer_id: hub node = anchor buyer, spokes =
    borrowers. Used for centrality-weighted contagion strength and for the
    dashboard's cluster network view."""
    g = nx.Graph()
    for anchor_id, grp in df.groupby("anchor_buyer_id"):
        hub = f"BUYER::{anchor_id}"
        g.add_node(hub, node_type="anchor_buyer", cluster_id=grp["cluster_id"].iloc[0])
        for _, row in grp.iterrows():
            g.add_node(row["borrower_id"], node_type="borrower")
            g.add_edge(hub, row["borrower_id"])
    return g


def compute_cluster_stress(df: pd.DataFrame, pd_col: str = "pd_12m") -> pd.DataFrame:
    """Per anchor-buyer cluster: account count, average PD, and a 0-100
    stress index relative to the portfolio-wide average PD."""
    portfolio_avg_pd = max(df[pd_col].mean(), 1e-6)

    grouped = df.groupby("anchor_buyer_id").agg(
        cluster_id=("cluster_id", "first"),
        sector=("sector", "first"),
        geography=("geography", "first"),
        n_accounts=("borrower_id", "count"),
        avg_pd=(pd_col, "mean"),
        total_exposure=("outstanding_amount", "sum"),
    ).reset_index()

    grouped["stress_ratio_vs_portfolio"] = (grouped["avg_pd"] / portfolio_avg_pd).round(2)
    grouped["cluster_stress_index"] = (
        np.clip(grouped["stress_ratio_vs_portfolio"] / 3.0, 0, 1) * 100
    ).round(1)
    grouped["is_elevated"] = (
        (grouped["stress_ratio_vs_portfolio"] >= ELEVATED_STRESS_MULTIPLE)
        & (grouped["n_accounts"] >= MIN_CLUSTER_SIZE_FOR_ALERT)
    )
    return grouped.sort_values("cluster_stress_index", ascending=False).reset_index(drop=True)


def apply_contagion_overlay(df: pd.DataFrame, pd_col: str = "pd_12m") -> pd.DataFrame:
    """Adds `cluster_stress_index`, `in_elevated_cluster`, and
    `pd_12m_contagion_adjusted` — a capped upward nudge to a borrower's own
    PD when their cluster is under elevated, broad-based stress. The
    adjustment never more than CONTAGION_MAX_UPLIFT relative, and only
    applies to clusters large enough that the signal isn't a single
    borrower's own PD leaking back into "cluster" stress."""
    cluster_stats = compute_cluster_stress(df, pd_col=pd_col)
    lookup = cluster_stats.set_index("anchor_buyer_id")[["cluster_stress_index", "is_elevated"]]

    out = df.copy()
    out = out.merge(lookup, left_on="anchor_buyer_id", right_index=True, how="left")
    out["cluster_stress_index"] = out["cluster_stress_index"].fillna(0.0)
    out["in_elevated_cluster"] = out["is_elevated"].fillna(False)
    out = out.drop(columns=["is_elevated"])

    uplift = np.where(
        out["in_elevated_cluster"],
        np.clip(out["cluster_stress_index"] / 100.0, 0, 1) * CONTAGION_MAX_UPLIFT,
        0.0,
    )
    out["contagion_uplift_pct"] = (uplift * 100).round(2)
    out["pd_12m_contagion_adjusted"] = np.clip(out[pd_col] * (1 + uplift), 0, 1).round(4)
    return out


def get_cluster_alerts(df: pd.DataFrame, pd_col: str = "pd_12m", top_n: int = 10) -> list[dict]:
    """Portfolio-level cluster alert feed, in the spirit of the design
    brief's dashboard card:
    '>> CLUSTER ALERT: Embroidery-Bhagatalao co-movement index elevated;
     anchor-buyer GSTIN filing delay detected -- 11 linked accounts.'
    """
    cluster_stats = compute_cluster_stress(df, pd_col=pd_col)
    elevated = cluster_stats[cluster_stats["is_elevated"]].head(top_n)

    alerts = []
    for _, row in elevated.iterrows():
        alerts.append({
            "anchor_buyer_id": row["anchor_buyer_id"],
            "cluster_id": row["cluster_id"],
            "sector": row["sector"],
            "geography": row["geography"],
            "n_accounts": int(row["n_accounts"]),
            "avg_pd": round(float(row["avg_pd"]), 4),
            "cluster_stress_index": float(row["cluster_stress_index"]),
            "total_exposure": float(row["total_exposure"]),
            "message": (
                f"{row['cluster_id']} co-movement index elevated "
                f"({row['stress_ratio_vs_portfolio']:.1f}x portfolio average PD) — "
                f"{int(row['n_accounts'])} linked accounts share anchor buyer {row['anchor_buyer_id']}."
            ),
        })
    return alerts
