#!/usr/bin/env python3
"""Streamlit dashboard for the AI public-investor monitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()
OUTPUT = ROOT / CONFIG.get("project", {}).get("output_dir", "outputs")

st.set_page_config(page_title="AI Capex Monitor", layout="wide")
st.title("AI Public-Investor Monitor")
st.caption(
    "SEC financial baseline + confidence-weighted AI disclosures + economic coverage and capex-cohort sensitivities"
)


@st.cache_data(show_spinner=False)
def read_csv(name: str, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = OUTPUT / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for col in dates:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


scorecard = read_csv("public_investor_scorecard", ("period_end",))
sec = read_csv("sec_quarterly_financials", ("period_end",))
coverage = read_csv("ai_economic_coverage", ("period_end",))
puw = read_csv("paid_useful_work_scores", ("as_of_date",))
ecosystem = read_csv("ecosystem_summary", ("latest_period_end",))
manual = read_csv("manual_ai_metric_summary", ("as_of_date", "source_date"))
irr = read_csv("capex_cohort_irr", ("period_end",))
snippets = read_csv("filing_evidence_snippets", ("filing_date", "report_date"))

if scorecard.empty and sec.empty and manual.empty:
    st.error("No output data found. Run `python ai_monitor.py run --user-agent \"Name email\"` first.")
    st.stop()


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"${value / scale:,.1f}{suffix}"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.1%}"


def multiple(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}x"


def latest_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return values.iloc[-1] if not values.empty else np.nan


all_tickers = sorted(
    set(scorecard.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    | set(sec.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    | set(manual.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
)

page = st.sidebar.radio("View", ["Overview", "Company", "Evidence", "Methodology"])

if page == "Overview":
    if not ecosystem.empty:
        base_market = ecosystem[ecosystem.get("scenario", "") == "base"]
        if not base_market.empty:
            market = base_market.iloc[-1]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ecosystem signal", str(market.get("ecosystem_signal", "n/a")))
            k2.metric(
                "Downstream paid-useful-work score",
                "n/a" if pd.isna(market.get("downstream_paid_useful_work_score")) else f"{market.get('downstream_paid_useful_work_score'):.1f}/100",
            )
            k3.metric(
                "Capital-owner AI coverage",
                multiple(market.get("capital_owner_ai_economic_coverage", np.nan)),
            )
            k4.metric(
                "Capital-owner capex growth",
                pct(market.get("capital_owner_total_capex_growth_yoy", np.nan)),
            )
            st.caption(
                "The capex-growth figure is company-wide for configured capital owners; AI coverage uses only your AI-attributed inputs."
            )

    st.subheader("Latest scorecard")
    if scorecard.empty:
        st.info("The scorecard has no rows yet.")
    else:
        display = scorecard.copy()
        percent_columns = [
            "revenue_growth_yoy",
            "gross_profit_growth_yoy",
            "capex_growth_yoy",
            "gross_profit_growth_minus_capex_growth",
            "downstream_metric_yoy_growth",
            "paid_useful_work_completeness_base",
            "paid_useful_work_confidence_base",
            "ai_input_confidence_base",
            "cohort_irr_base",
        ]
        for col in percent_columns:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce") * 100.0
        preferred = [
            "ticker",
            "company",
            "layer",
            "revenue_growth_yoy",
            "gross_profit_growth_yoy",
            "capex_growth_yoy",
            "downstream_metric",
            "downstream_metric_yoy_growth",
            "paid_useful_work_score_base",
            "paid_useful_work_status_base",
            "paid_useful_work_completeness_base",
            "ai_economic_coverage_base",
            "catchup_status_base",
            "cohort_irr_base",
            "cohort_irr_status_base",
            "ai_input_confidence_base",
            "manual_metric_count",
        ]
        preferred = [c for c in preferred if c in display.columns]
        st.dataframe(
            display[preferred].sort_values(["catchup_status_base", "ticker"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "revenue_growth_yoy": st.column_config.NumberColumn(format="%.1f%%"),
                "gross_profit_growth_yoy": st.column_config.NumberColumn(format="%.1f%%"),
                "capex_growth_yoy": st.column_config.NumberColumn(format="%.1f%%"),
                "downstream_metric_yoy_growth": st.column_config.NumberColumn(format="%.1f%%"),
                "paid_useful_work_score_base": st.column_config.NumberColumn(format="%.1f"),
                "paid_useful_work_completeness_base": st.column_config.NumberColumn(format="%.0f%%"),
                "ai_economic_coverage_base": st.column_config.NumberColumn(format="%.2fx"),
                "cohort_irr_base": st.column_config.NumberColumn(format="%.1f%%"),
                "ai_input_confidence_base": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
        st.caption(
            "Streamlit displays decimals as percentages only when the underlying value is a fraction; CSV exports retain raw decimal values."
        )

    if not sec.empty:
        latest_sec = sec.sort_values("period_end").groupby("ticker", as_index=False).tail(1)
        chart_cols = [c for c in ["gross_profit_ttm_yoy", "capex_ttm_yoy"] if c in latest_sec.columns]
        if len(chart_cols) == 2:
            plot = latest_sec[["ticker"] + chart_cols].melt(
                id_vars="ticker", var_name="metric", value_name="growth"
            )
            fig = px.bar(plot, x="ticker", y="growth", barmode="group", facet_col=None, title="TTM gross-profit growth versus capex growth")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Company":
    if not all_tickers:
        st.info("No companies are available.")
        st.stop()
    ticker = st.sidebar.selectbox("Company", all_tickers)
    sec_t = sec[sec.get("ticker", "") == ticker].sort_values("period_end") if not sec.empty else pd.DataFrame()
    cov_t = coverage[coverage.get("ticker", "") == ticker].sort_values("period_end") if not coverage.empty else pd.DataFrame()
    man_t = manual[manual.get("ticker", "") == ticker].sort_values("as_of_date") if not manual.empty else pd.DataFrame()
    irr_t = irr[irr.get("ticker", "") == ticker] if not irr.empty else pd.DataFrame()
    puw_t = puw[puw.get("ticker", "") == ticker].sort_values("as_of_date") if not puw.empty else pd.DataFrame()

    title_company = ticker
    if not sec_t.empty and "company" in sec_t.columns:
        title_company = f"{ticker} — {sec_t['company'].iloc[-1]}"
    elif not man_t.empty and "company" in man_t.columns and pd.notna(man_t["company"].iloc[-1]):
        title_company = f"{ticker} — {man_t['company'].iloc[-1]}"
    st.subheader(title_company)

    base_cov = cov_t[cov_t.get("scenario", "") == "base"] if not cov_t.empty else pd.DataFrame()
    base_puw = puw_t[puw_t.get("scenario", "") == "base"] if not puw_t.empty else pd.DataFrame()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Revenue TTM", money(latest_value(sec_t, "revenue_ttm")))
    c2.metric("Gross-profit growth", pct(latest_value(sec_t, "gross_profit_ttm_yoy")))
    c3.metric("Capex growth", pct(latest_value(sec_t, "capex_ttm_yoy")))
    c4.metric(
        "Paid-useful-work score",
        "n/a" if pd.isna(latest_value(base_puw, "paid_useful_work_score")) else f"{latest_value(base_puw, 'paid_useful_work_score'):.1f}/100",
    )
    c5.metric("AI economic coverage", multiple(latest_value(base_cov, "ai_economic_coverage")))
    c6.metric("Base cohort IRR", pct(latest_value(irr_t[irr_t.get("scenario", "") == "base"] if not irr_t.empty else pd.DataFrame(), "cohort_irr")))

    if not sec_t.empty:
        available = [c for c in ["revenue_ttm", "gross_profit_ttm", "capex_ttm", "operating_cash_flow_ttm"] if c in sec_t.columns]
        if available:
            plot = sec_t[["period_end"] + available].melt(
                id_vars="period_end", var_name="metric", value_name="value"
            ).dropna()
            fig = px.line(plot, x="period_end", y="value", line_dash="metric", markers=True, title="SEC financial baseline — trailing twelve months")
            st.plotly_chart(fig, use_container_width=True)

        growth_cols = [c for c in ["gross_profit_ttm_yoy", "capex_ttm_yoy", "operating_cash_flow_ttm_yoy"] if c in sec_t.columns]
        if growth_cols:
            plot = sec_t[["period_end"] + growth_cols].melt(
                id_vars="period_end", var_name="metric", value_name="growth"
            ).dropna()
            fig = px.line(plot, x="period_end", y="growth", line_dash="metric", markers=True, title="Growth comparison")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    if not puw_t.empty and puw_t["paid_useful_work_score"].notna().any():
        latest_puw = puw_t.sort_values("as_of_date").groupby("scenario", as_index=False).tail(1)
        st.subheader("Paid useful work proxy")
        st.dataframe(
            latest_puw[
                [
                    c
                    for c in [
                        "scenario",
                        "paid_useful_work_score",
                        "puw_status",
                        "puw_data_completeness",
                        "puw_confidence",
                        "paid_demand_source_metric",
                        "paid_demand_value",
                        "production_value",
                        "success_value",
                        "persistence_value",
                        "margin_value",
                    ]
                    if c in latest_puw.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not cov_t.empty and cov_t["ai_economic_coverage"].notna().any():
        fig = px.line(
            cov_t.dropna(subset=["ai_economic_coverage"]),
            x="period_end",
            y="ai_economic_coverage",
            line_dash="scenario",
            markers=True,
            title="Downstream contribution profit / AI economic capital charge",
        )
        st.plotly_chart(fig, use_container_width=True)
        latest_cov = cov_t.sort_values("period_end").groupby("scenario", as_index=False).tail(1)
        st.dataframe(
            latest_cov[
                [
                    c
                    for c in [
                        "scenario",
                        "ai_economic_coverage",
                        "ai_contribution_return_on_net_capital",
                        "ai_reinvestment_coverage",
                        "ai_input_confidence",
                        "ai_coverage_basis",
                        "catchup_status",
                        "ai_input_method",
                    ]
                    if c in latest_cov.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "AI-specific coverage is unavailable until manual_ai_metrics.csv contains an AI capex estimate and an AI contribution-profit or AI revenue/margin estimate."
        )

    if not man_t.empty:
        st.subheader("Latest manual AI disclosures")
        st.dataframe(
            man_t[
                [
                    c
                    for c in [
                        "metric",
                        "scenario",
                        "as_of_date",
                        "value",
                        "unit",
                        "period_type",
                        "confidence",
                        "yoy_growth",
                        "source_name",
                        "reported_or_estimated",
                        "notes",
                    ]
                    if c in man_t.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

elif page == "Evidence":
    st.subheader("Recent filing evidence snippets")
    if snippets.empty:
        st.info("Run the pipeline with `--with-snippets` to populate this page.")
    else:
        tickers = ["All"] + sorted(snippets["ticker"].dropna().astype(str).unique().tolist())
        ticker = st.sidebar.selectbox("Ticker", tickers)
        groups = ["All"] + sorted(snippets["keyword_group"].dropna().astype(str).unique().tolist())
        group = st.sidebar.selectbox("Keyword group", groups)
        view = snippets.copy()
        if ticker != "All":
            view = view[view["ticker"] == ticker]
        if group != "All":
            view = view[view["keyword_group"] == group]
        st.dataframe(
            view.sort_values("filing_date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={"source_url": st.column_config.LinkColumn("Source")},
        )

else:
    st.subheader("How to interpret the monitor")
    st.markdown(
        """
The monitor deliberately separates three questions:

1. **Is paid downstream usage growing?** Manual disclosures capture AI revenue, ARR, paid useful-work indices, production share, renewals, customer counts, and similar evidence.
2. **Is downstream contribution profit covering the capital charge?** Economic coverage divides estimated AI contribution profit by economic depreciation, the required return on AI capital, and maintenance capital.
3. **Could a new capex cohort clear the investment hurdle?** The cohort model tests an explicit revenue-per-capital, utilization, margin, useful-life, price, and efficiency scenario.

A result of **INSUFFICIENT_AI_DISCLOSURE** is intentional. It is better than assigning false precision where public reporting does not identify AI revenue, AI costs, or the portion of capex attributable to AI.

Edit these files before relying on the result:

- `data/manual_ai_metrics.csv`
- `data/cohort_assumptions.csv`
- `config.yaml`

The SEC baseline is company-wide. AI attribution remains an investor estimate unless management explicitly reports it.
        """
    )
