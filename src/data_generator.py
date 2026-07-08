"""
Bharat MSME Credit Radar - Synthetic Data Generator
=====================================================
Generates realistic, borrower-month level synthetic data resembling Indian
MSME lending books: borrower profile, repayment conduct, GST returns,
bank-statement / Account Aggregator cash-flow signals, bureau data, EPFO
employment signals, and free-text credit officer remarks.

The data is entirely synthetic. It is built around a hidden per-borrower
"latent risk propensity" so that the generated features are internally
consistent (a borrower drifting toward stress shows deteriorating DPD,
bounces, GST mismatches, cash-flow strain, falling bureau score, etc. at
the same time) and so that the eventual `stress_12m` target is learnable
from the features, as would be expected of a real MSME lending book.

Run:
    python src/data_generator.py --n-borrowers 3500 --months 8 --out data/synthetic_msme_data.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RNG_SEED = 42

SEGMENTS = ["Trader", "Manufacturer", "Service", "NTC", "Existing", "CGTMSE", "Thin-file"]
CONSTITUTIONS = ["Proprietorship", "Partnership", "Pvt Ltd", "LLP"]
SECTORS = [
    "Textile", "Engineering", "Chemicals", "Food Processing",
    "Gems & Jewellery", "Services", "Retail", "Construction",
]
GEOGRAPHIES = [
    "Surat", "Rajkot", "Vadodara", "Ahmedabad", "Mumbai",
    "Pune", "Jaipur", "Ludhiana", "Coimbatore",
]
LOAN_TYPES = ["Cash Credit", "Term Loan", "Machinery Loan", "LAP", "CGTMSE", "Working Capital"]

# Baseline relative riskiness used only to shape the synthetic latent risk factor.
SEGMENT_RISK = {"Trader": 0.10, "Manufacturer": 0.0, "Service": -0.05, "NTC": 0.55,
                "Existing": -0.25, "CGTMSE": 0.05, "Thin-file": 0.45}
SECTOR_RISK = {"Textile": 0.20, "Engineering": 0.0, "Chemicals": 0.05, "Food Processing": -0.10,
               "Gems & Jewellery": 0.35, "Services": -0.10, "Retail": 0.10, "Construction": 0.30}
GEOGRAPHY_RISK = {"Surat": 0.10, "Rajkot": 0.05, "Vadodara": 0.0, "Ahmedabad": -0.05,
                   "Mumbai": -0.10, "Pune": -0.10, "Jaipur": 0.05, "Ludhiana": 0.15,
                   "Coimbatore": -0.05}
CONSTITUTION_RISK = {"Proprietorship": 0.15, "Partnership": 0.05, "Pvt Ltd": -0.15, "LLP": -0.05}

POSITIVE_REMARKS = {
    "cam": ["business running satisfactorily", "financials in line with projections",
            "promoter conduct satisfactory", "adequate liquidity maintained",
            "repayment track record satisfactory"],
    "fi": ["business premises verified and active", "stock levels found adequate",
           "field investigation found no adverse observation", "unit operational with visible activity"],
    "rcu": ["documents verified genuine", "no adverse RCU observation", "identity and address cross-verified",
            "no linkage with other defaulting entities found"],
    "collection": ["repayment conduct satisfactory", "borrower proactively regularised dues",
                   "no follow up required this month", "EMI realised on due date"],
    "stock": ["stock verified and found adequate", "stock statement matches book records",
              "inventory levels healthy and moving", "stock audit found no discrepancy"],
}
NEGATIVE_REMARKS = {
    "cam": ["cash flow stress visible", "buyer base concentrated with top two buyers",
            "declining turnover trend observed", "related party transactions suspected",
            "promoter diverting funds to group entity suspected"],
    "fi": ["shop found closed during visit", "frequent promise to pay noted",
           "unit operations appear scaled down", "technical report indicates property access issue"],
    "rcu": ["RCU noted common mobile number with other entities", "address verification inconclusive",
            "discrepancy noted in KYC documents", "common email id linked to multiple applicants"],
    "collection": ["frequent promise to pay", "borrower unreachable on registered number",
                   "part payment received after repeated follow up", "cheque returned for insufficient funds"],
    "stock": ["stock not matching book records", "stock ageing beyond acceptable limits",
              "obsolete stock observed during inspection", "stock verification could not be completed"],
}


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_borrowers(n_borrowers: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per unique borrower with static profile + latent risk factor."""
    borrower_id = [f"B{100000 + i}" for i in range(n_borrowers)]
    borrower_name = [f"MSME Enterprise {i:05d}" for i in range(n_borrowers)]

    segment = rng.choice(SEGMENTS, size=n_borrowers, p=[0.20, 0.20, 0.18, 0.10, 0.15, 0.10, 0.07])
    constitution = rng.choice(CONSTITUTIONS, size=n_borrowers, p=[0.45, 0.20, 0.25, 0.10])
    sector = rng.choice(SECTORS, size=n_borrowers)
    geography = rng.choice(GEOGRAPHIES, size=n_borrowers)

    business_vintage_years = _clip(rng.gamma(shape=3.0, scale=2.2, size=n_borrowers), 0.5, 35).round(1)
    loan_type = rng.choice(LOAN_TYPES, size=n_borrowers, p=[0.30, 0.20, 0.12, 0.10, 0.13, 0.15])

    sanctioned_limit = _clip(rng.lognormal(mean=14.7, sigma=0.9, size=n_borrowers), 3e5, 5e7).round(0)
    util_seed = _clip(rng.beta(2.2, 2.0, size=n_borrowers), 0.05, 1.05)
    outstanding_amount = (sanctioned_limit * util_seed).round(0)

    collateral_available = rng.choice(["Yes", "No"], size=n_borrowers, p=[0.55, 0.45])
    cgtmse_flag = np.where(segment == "CGTMSE", "Yes",
                            rng.choice(["Yes", "No"], size=n_borrowers, p=[0.12, 0.88]))

    latent = (
        pd.Series(segment).map(SEGMENT_RISK).to_numpy()
        + pd.Series(sector).map(SECTOR_RISK).to_numpy()
        + pd.Series(geography).map(GEOGRAPHY_RISK).to_numpy()
        + pd.Series(constitution).map(CONSTITUTION_RISK).to_numpy()
        + (-0.03 * business_vintage_years)
        + np.where(collateral_available == "No", 0.30, -0.10)
        + rng.normal(0, 0.65, size=n_borrowers)
    )
    latent = (latent - latent.mean()) / latent.std()

    # Cluster / anchor-buyer structure for the graph contagion overlay: every
    # borrower belongs to a sector-geography cluster cell (e.g. "Textile-Surat"),
    # further split into a handful of anchor-buyer groups within that cell —
    # modelling the real MSME-cluster pattern where many small units in the same
    # trade pocket sell to a small number of anchor buyers/traders.
    cluster_id = pd.Series(sector).astype(str) + "-" + pd.Series(geography).astype(str)
    n_sub_buyers = rng.integers(2, 4, size=n_borrowers)  # 2 or 3 anchor buyers per cluster cell
    anchor_buyer_id = cluster_id + "-BUYER" + pd.Series(n_sub_buyers).astype(str)

    df = pd.DataFrame({
        "borrower_id": borrower_id,
        "borrower_name": borrower_name,
        "segment": segment,
        "constitution": constitution,
        "sector": sector,
        "geography": geography,
        "business_vintage_years": business_vintage_years,
        "loan_type": loan_type,
        "sanctioned_limit": sanctioned_limit,
        "outstanding_amount": outstanding_amount,
        "collateral_available": collateral_available,
        "CGTMSE_flag": cgtmse_flag,
        "cluster_id": cluster_id,
        "anchor_buyer_id": anchor_buyer_id,
        "_latent_risk": latent,
    })
    return df


def add_cluster_contagion_shock(rep: pd.DataFrame, months: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate anchor-buyer / cluster contagion: a minority of anchor-buyer
    groups become distressed (e.g. an anchor trader/buyer starts struggling)
    and that shock ramps up across months, lifting `_risk_now` for every
    borrower who sells into that buyer group — independent of each
    borrower's own individual conduct. This is what the graph contagion
    overlay (src/graph_contagion.py) is designed to detect early from
    co-movement, before each member's own signals fully deteriorate."""
    anchor_ids = rep["anchor_buyer_id"].unique()
    n_distressed = max(1, int(round(len(anchor_ids) * 0.08)))
    distressed = set(rng.choice(anchor_ids, size=n_distressed, replace=False))
    rep["_anchor_distressed"] = rep["anchor_buyer_id"].isin(distressed).astype(int)

    # Ramp severity is anchor-specific (not every distressed buyer fails equally hard).
    severity = {a: rng.uniform(0.6, 1.8) for a in distressed}
    sev = rep["anchor_buyer_id"].map(severity).fillna(0.0).to_numpy()
    ramp = rep["obs_month"].to_numpy() / max(months - 1, 1)
    rep["_cluster_shock"] = sev * ramp * rep["_anchor_distressed"].to_numpy()
    return rep


def expand_panel(borrowers: pd.DataFrame, months: int, rng: np.random.Generator) -> pd.DataFrame:
    """Replicate each borrower across `months` observation months, adding a
    per-row stress trajectory that drifts upward for higher latent-risk
    borrowers (simulating deterioration approaching a stress event), plus a
    shared anchor-buyer cluster shock for a minority of borrower clusters."""
    n = len(borrowers)
    rep = borrowers.loc[borrowers.index.repeat(months)].reset_index(drop=True)
    month_idx = np.tile(np.arange(months), n)
    rep["obs_month"] = month_idx
    rep = add_cluster_contagion_shock(rep, months, rng)

    # Trajectory noise: each borrower gets its own drift slope correlated with latent risk.
    drift_slope = rep["_latent_risk"].to_numpy() * rng.normal(1.0, 0.25, size=len(rep))
    month_noise = rng.normal(0, 0.35, size=len(rep))
    rep["_risk_now"] = (
        rep["_latent_risk"] + drift_slope * (rep["obs_month"] / max(months - 1, 1))
        + month_noise + rep["_cluster_shock"]
    )
    return rep


def add_repayment_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    dpd_lambda = _clip(np.exp(1.35 * z - 1.6), 0.01, 150)
    current_dpd = rng.poisson(dpd_lambda)
    current_dpd = _clip(current_dpd - rng.integers(0, 2, size=n), 0, 180)
    max_dpd_last_12m = _clip(current_dpd + rng.poisson(_clip(2 + 6 * np.maximum(z, 0), 0, 40)), 0, 180)

    df["current_dpd"] = current_dpd
    df["max_dpd_last_12m"] = max_dpd_last_12m
    df["emi_bounce_count_6m"] = rng.poisson(_clip(np.exp(0.6 * z - 0.4), 0, 8)).astype(int)
    df["cheque_return_count_6m"] = rng.poisson(_clip(np.exp(0.55 * z - 0.7), 0, 6)).astype(int)
    df["si_ecs_bounce_count_6m"] = rng.poisson(_clip(np.exp(0.6 * z - 0.6), 0, 7)).astype(int)

    df["repayment_regularity_score"] = _clip(
        100 - 14 * np.maximum(z, -1) - df["emi_bounce_count_6m"] * 3 + rng.normal(0, 4, n), 0, 100
    ).round(1)

    util_center = _clip(0.55 + 0.16 * z, 0.05, 1.05)
    df["cc_utilization_avg_3m"] = (_clip(rng.normal(util_center, 0.12, n), 0, 1.15) * 100).round(1)
    df["cc_utilization_avg_6m"] = (_clip(rng.normal(util_center - 0.03, 0.13, n), 0, 1.15) * 100).round(1)
    df["cc_utilization_volatility"] = _clip(rng.normal(6 + 6 * np.maximum(z, 0), 3, n), 0.5, 40).round(2)
    df["drawing_power_decline_pct"] = _clip(rng.normal(3 + 9 * np.maximum(z, 0), 5, n), -10, 80).round(1)
    df["overdue_days_trend"] = _clip(rng.normal(2 * z, 3, n), -15, 60).round(1)

    def sma(dpd):
        if dpd <= 0:
            return "Standard"
        if dpd <= 30:
            return "SMA-0"
        if dpd <= 60:
            return "SMA-1"
        if dpd <= 90:
            return "SMA-2"
        return "NPA"

    df["SMA_status"] = df["current_dpd"].apply(sma)
    return df


def add_gst_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    base_turnover = df["sanctioned_limit"] * rng.uniform(1.2, 3.5, n)
    df["gst_turnover_12m"] = base_turnover.round(0)
    df["gst_turnover_growth_yoy"] = _clip(rng.normal(8 - 14 * np.maximum(z, 0), 10, n), -70, 90).round(1)
    df["gst_filing_delay_count_6m"] = rng.poisson(_clip(np.exp(0.55 * z - 0.9), 0, 6)).astype(int)
    df["gstr1_vs_3b_mismatch_pct"] = _clip(rng.normal(3 + 9 * np.maximum(z, 0), 4, n), 0, 60).round(1)
    df["itc_to_sales_ratio"] = _clip(rng.normal(0.55 + 0.15 * np.maximum(z, 0), 0.12, n), 0.05, 1.3).round(3)
    df["nil_return_count_12m"] = rng.poisson(_clip(0.4 + 0.8 * np.maximum(z, 0), 0, 6)).astype(int)
    df["sudden_turnover_spike_flag"] = (rng.uniform(0, 1, n) < _clip(0.03 + 0.06 * np.maximum(z, 0), 0, 0.4)).astype(int)
    df["gst_registration_age_years"] = _clip(df["business_vintage_years"] - rng.uniform(0, 1.5, n), 0.2, 35).round(1)

    gst_status_p_cancel = _clip(0.01 + 0.05 * np.maximum(z, 0), 0.005, 0.35)
    gst_status_p_suspend = _clip(0.02 + 0.06 * np.maximum(z, 0), 0.01, 0.30)
    u = rng.uniform(0, 1, n)
    df["gst_status"] = np.where(u < gst_status_p_cancel, "Cancelled",
                        np.where(u < gst_status_p_cancel + gst_status_p_suspend, "Suspended", "Active"))

    df["buyer_concentration_top2_pct"] = _clip(rng.normal(35 + 18 * np.maximum(z, 0), 12, n), 8, 98).round(1)
    df["supplier_concentration_top2_pct"] = _clip(rng.normal(32 + 15 * np.maximum(z, 0), 12, n), 8, 98).round(1)
    df["eway_bill_mismatch_flag"] = (rng.uniform(0, 1, n) < _clip(0.03 + 0.08 * np.maximum(z, 0), 0, 0.5)).astype(int)
    return df


def add_bank_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    monthly_gst_sales = df["gst_turnover_12m"] / 12
    credit_ratio = _clip(rng.normal(0.85 - 0.18 * np.maximum(z, 0), 0.15, n), 0.15, 1.4)
    df["avg_monthly_bank_credit_6m"] = (monthly_gst_sales * credit_ratio).round(0)
    df["avg_monthly_bank_debit_6m"] = (df["avg_monthly_bank_credit_6m"] * _clip(rng.normal(0.92, 0.08, n), 0.5, 1.3)).round(0)
    df["bank_credit_to_gst_sales_ratio"] = (df["avg_monthly_bank_credit_6m"] / monthly_gst_sales.replace(0, np.nan)).round(3).fillna(0.5)

    df["cash_deposit_ratio"] = _clip(rng.normal(0.18 + 0.20 * np.maximum(z, 0), 0.10, n), 0.02, 0.95).round(3)
    df["avg_monthly_balance"] = _clip(df["avg_monthly_bank_credit_6m"] * rng.normal(0.06 - 0.02 * np.maximum(z, 0), 0.03, n), 500, None).round(0)
    df["inward_return_count_6m"] = rng.poisson(_clip(np.exp(0.5 * z - 1.0), 0, 6)).astype(int)
    df["outward_return_count_6m"] = rng.poisson(_clip(np.exp(0.5 * z - 1.1), 0, 6)).astype(int)
    df["upi_pos_collection_ratio"] = _clip(rng.normal(0.45 - 0.15 * np.maximum(z, 0), 0.15, n), 0.02, 0.95).round(3)
    df["cashflow_volatility_score"] = _clip(rng.normal(25 + 22 * np.maximum(z, 0), 10, n), 2, 100).round(1)
    df["debt_service_coverage_proxy"] = _clip(rng.normal(1.35 - 0.45 * np.maximum(z, 0), 0.35, n), 0.1, 3.5).round(2)
    df["monthly_surplus_ratio"] = _clip(rng.normal(0.12 - 0.18 * np.maximum(z, 0), 0.10, n), -0.5, 0.6).round(3)
    return df


def add_bureau_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    df["bureau_score"] = _clip(rng.normal(725 - 95 * np.maximum(z, 0), 45, n), 300, 900).round(0).astype(int)
    df["bureau_enquiry_count_3m"] = rng.poisson(_clip(1 + 2.4 * np.maximum(z, 0), 0, 20)).astype(int)
    df["existing_loan_count"] = rng.poisson(_clip(1.5 + 0.6 * np.maximum(z, 0), 0, 12)).astype(int)
    df["total_obligation"] = (df["outstanding_amount"] * _clip(rng.normal(1.3, 0.5, n), 0.2, 4)).round(0)
    df["unsecured_loan_exposure"] = (df["total_obligation"] * _clip(rng.normal(0.20 + 0.20 * np.maximum(z, 0), 0.12, n), 0, 0.95)).round(0)
    df["bureau_dpd_last_12m"] = _clip(df["max_dpd_last_12m"] + rng.poisson(2, n) - 1, 0, 180)
    df["credit_vintage_years"] = _clip(df["business_vintage_years"] * rng.uniform(0.6, 1.0, n), 0.3, 35).round(1)
    return df


def add_epfo_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    base_emp = _clip(rng.lognormal(mean=2.3, sigma=0.9, size=n), 1, 500)
    df["epfo_employee_count"] = base_emp.round(0).astype(int)
    df["epfo_employee_count_change_6m"] = _clip(rng.normal(2 - 14 * np.maximum(z, 0), 8, n), -90, 60).round(1)
    df["salary_payment_regularity_score"] = _clip(rng.normal(92 - 20 * np.maximum(z, 0), 8, n), 10, 100).round(1)
    return df


def add_text_vars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = df["_risk_now"].to_numpy()
    n = len(df)

    def make_remark(remark_type, neg_prob):
        pos_pool = POSITIVE_REMARKS[remark_type]
        neg_pool = NEGATIVE_REMARKS[remark_type]
        out = []
        for p in neg_prob:
            n_neg = rng.binomial(2, _clip(p, 0, 0.95))
            n_pos = rng.integers(0, 2)
            sentences = list(rng.choice(neg_pool, size=n_neg, replace=False)) if n_neg else []
            if n_pos:
                sentences += list(rng.choice(pos_pool, size=min(n_pos, len(pos_pool)), replace=False))
            if not sentences:
                sentences = [rng.choice(pos_pool)]
            rng.shuffle(sentences)
            out.append("; ".join(sentences) + ".")
        return out

    neg_prob = _clip(0.10 + 0.28 * np.maximum(z, 0), 0.02, 0.9)
    df["cam_remarks"] = make_remark("cam", neg_prob)
    df["fi_remarks"] = make_remark("fi", neg_prob)
    df["rcu_remarks"] = make_remark("rcu", neg_prob * 0.7)
    df["collection_remarks"] = make_remark("collection", neg_prob)
    df["stock_inspection_remarks"] = make_remark("stock", neg_prob)
    return df


def compute_target(df: pd.DataFrame, rng: np.random.Generator, target_rate: float = 0.065) -> pd.DataFrame:
    """Binary stress_12m built from a logistic combination of the strongest
    stress indicators, then intercept-calibrated (bisection) to hit the
    desired overall prevalence (~5-8%), matching a realistic MSME book."""
    n = len(df)

    # Note: deliberately built from *realized, noisy* observable signals rather
    # than the hidden latent risk factor directly, and blended with a sizeable
    # idiosyncratic noise term (unmodelled shocks: fraud, macro, one-off buyer
    # default, etc.) so the resulting target is learnable but not trivially/
    # perfectly separable -- mirroring how real MSME stress events retain a
    # genuinely unpredictable component even with rich alternate data.
    score = (
        0.028 * df["current_dpd"].to_numpy()
        + 0.30 * df["emi_bounce_count_6m"].to_numpy()
        + 0.26 * df["cheque_return_count_6m"].to_numpy()
        + 0.016 * df["gstr1_vs_3b_mismatch_pct"].to_numpy()
        + 0.012 * df["drawing_power_decline_pct"].to_numpy()
        - 0.0045 * (df["bureau_score"].to_numpy() - 700)
        + 0.014 * df["buyer_concentration_top2_pct"].to_numpy()
        - 0.55 * df["debt_service_coverage_proxy"].to_numpy()
        + 0.010 * df["cashflow_volatility_score"].to_numpy()
        + 0.10 * df["gst_filing_delay_count_6m"].to_numpy()
        - 0.05 * df["epfo_employee_count_change_6m"].to_numpy()
        + rng.normal(0, 1.8, size=n)
    )

    def prevalence(intercept):
        p = 1 / (1 + np.exp(-(score + intercept)))
        return p.mean()

    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if prevalence(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2

    prob = 1 / (1 + np.exp(-(score + intercept)))
    df["stress_12m"] = (rng.uniform(0, 1, size=len(df)) < prob).astype(int)
    return df


def compute_growth_target(df: pd.DataFrame, rng: np.random.Generator, target_rate: float = 0.14) -> pd.DataFrame:
    """Binary growth_need_12m target for the MSME Growth Propensity Engine:
    the "mirror image" of compute_target, built from expansion/headroom
    signals rather than stress signals — sustained GST growth, rising EPFO
    headcount, comfortable debt-service coverage, positive cash surplus and
    persistently high (but currently serviced) CC utilisation (a unit
    outrunning its limit). Intercept-calibrated to a ~12-16% prevalence,
    reflecting that meaningfully more accounts show *some* growth signal
    than show stress in any given year. Eligibility (Green/Yellow grade,
    clean fraud/authenticity) is enforced separately at scoring time by
    src/growth_propensity.py — this target represents the underlying
    "would benefit from an enhanced facility" state, independent of whether
    the Radar currently trusts the account enough to act on it."""
    n = len(df)
    score = (
        0.045 * df["gst_turnover_growth_yoy"].to_numpy()
        + 0.035 * df["epfo_employee_count_change_6m"].to_numpy()
        + 1.10 * df["debt_service_coverage_proxy"].to_numpy()
        + 0.020 * df["cc_utilization_avg_3m"].to_numpy()
        + 3.20 * df["monthly_surplus_ratio"].to_numpy()
        - 0.018 * df["buyer_concentration_top2_pct"].to_numpy()
        - 0.05 * df["current_dpd"].to_numpy()
        - 0.35 * df["emi_bounce_count_6m"].to_numpy()
        + 0.012 * df["upi_pos_collection_ratio"].to_numpy() * 100
        + rng.normal(0, 1.6, size=n)
    )

    def prevalence(intercept):
        p = 1 / (1 + np.exp(-(score + intercept)))
        return p.mean()

    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if prevalence(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2

    prob = 1 / (1 + np.exp(-(score + intercept)))
    df["growth_need_12m"] = (rng.uniform(0, 1, size=len(df)) < prob).astype(int)
    return df


def generate(n_borrowers: int, months: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    borrowers = generate_borrowers(n_borrowers, rng)
    panel = expand_panel(borrowers, months, rng)

    panel = add_repayment_vars(panel, rng)
    panel = add_gst_vars(panel, rng)
    panel = add_bank_vars(panel, rng)
    panel = add_bureau_vars(panel, rng)
    panel = add_epfo_vars(panel, rng)
    panel = add_text_vars(panel, rng)
    panel = compute_target(panel, rng)
    panel = compute_growth_target(panel, rng)

    panel = panel.drop(columns=["_latent_risk", "_risk_now", "_anchor_distressed", "_cluster_shock"])
    panel.insert(0, "record_id", [f"REC{i:07d}" for i in range(len(panel))])
    return panel


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Bharat MSME Credit Radar dataset.")
    parser.add_argument("--n-borrowers", type=int, default=3500)
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--out", type=str, default=os.path.join("data", "synthetic_msme_data.csv"))
    args = parser.parse_args()

    df = generate(args.n_borrowers, args.months, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df):,} borrower-month rows for {args.n_borrowers:,} borrowers.")
    print(f"Stress rate: {df['stress_12m'].mean():.2%}")
    print(f"Growth-need rate: {df['growth_need_12m'].mean():.2%}")
    print(f"Cluster cells: {df['cluster_id'].nunique()}  Anchor-buyer groups: {df['anchor_buyer_id'].nunique()}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
