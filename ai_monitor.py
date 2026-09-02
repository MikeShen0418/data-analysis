#!/usr/bin/env python3
"""Public-investor AI capex monitor.

The pipeline combines:
1. Automatically downloaded SEC Company Facts for public companies.
2. Optional SEC filing keyword snippets for qualitative evidence.
3. Manually entered AI disclosures for public and private companies.
4. A transparent AI capital-stock and economic-coverage model.
5. Optional capex-cohort IRR sensitivities.

The goal is not to pretend that private AI economics are fully observable. The
output explicitly separates reported, derived, and modeled values and carries
source confidence through the analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    duckdb = None

try:
    import numpy_financial as npf  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    npf = None


APP_VERSION = "1.0.0"
ACCEPTED_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
SCENARIOS = ("downside", "base", "upside")

# XBRL concept aliases. The first tag is preferred when several tags are
# available for the same period, but the parser will fill gaps from later tags.
FLOW_METRICS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "Depreciation",
    ],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "share_based_compensation": ["ShareBasedCompensation"],
}

INSTANT_METRICS: dict[str, list[str]] = {
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
    ],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "total_assets": ["Assets"],
}

# Metrics for which a higher value improves the AI investment case. For metrics
# not listed here, the scenario mapping assumes higher values are adverse.
HIGH_IS_GOOD = {
    "ai_revenue",
    "ai_revenue_run_rate",
    "ai_arr",
    "ai_rpo",
    "ai_contribution_profit",
    "ai_contribution_profit_run_rate",
    "ai_contribution_margin",
    "paid_useful_work_index",
    "paid_token_index",
    "production_share",
    "renewal_rate",
    "net_revenue_retention",
    "paid_enterprise_customers",
    "task_success_rate",
    "customer_value_index",
    "gpu_utilization",
    "spot_gpu_price_index",
    "ai_incremental_gp_attribution_share",
    "inference_share",
}

MANUAL_COLUMNS = [
    "as_of_date",
    "ticker",
    "company",
    "metric",
    "value",
    "low",
    "high",
    "unit",
    "period_type",
    "source_name",
    "source_url",
    "source_date",
    "confidence",
    "reported_or_estimated",
    "notes",
]

ASSUMPTION_COLUMNS = [
    "ticker",
    "scenario",
    "hurdle_rate",
    "ai_asset_life_years",
    "maintenance_capex_pct",
    "tax_rate",
    "revenue_per_capital_at_full_utilization",
    "contribution_margin",
    "year1_utilization",
    "year2_utilization",
    "mature_utilization",
    "annual_price_change",
    "annual_efficiency_capture",
    "salvage_pct",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def safe_div(numerator: Any, denominator: Any) -> Any:
    """Vector-friendly division that returns NaN for zero/invalid denominators."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        n = pd.to_numeric(numerator, errors="coerce")
        d = pd.to_numeric(denominator, errors="coerce")
        return n.div(d.where(d.abs() > 1e-12))
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    return n / d if math.isfinite(d) and abs(d) > 1e-12 else np.nan


def first_non_null(values: Iterable[Any]) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return np.nan


def annualize_flow(value: float, period_type: str) -> float:
    kind = (period_type or "").strip().lower()
    if kind in {"quarter", "quarterly", "q"}:
        return value * 4.0
    if kind in {"month", "monthly"}:
        return value * 12.0
    if kind in {"half_year", "semiannual", "six_months"}:
        return value * 2.0
    # annual, TTM, run-rate, and point-in-time monetary run rates are already
    # interpreted as annualized. Point-in-time ratios are unaffected.
    return value


def quarterly_flow(value: float, period_type: str) -> float:
    kind = (period_type or "").strip().lower()
    if kind in {"quarter", "quarterly", "q"}:
        return value
    if kind in {"month", "monthly"}:
        return value * 3.0
    if kind in {"half_year", "semiannual", "six_months"}:
        return value / 2.0
    if kind in {"annual", "year", "yearly", "ttm", "run_rate", "annualized"}:
        return value / 4.0
    return value


def normalize_ratio(value: float) -> float:
    """Convert percentages entered as 35 to 0.35 while preserving 0.35."""
    if pd.isna(value):
        return np.nan
    value = float(value)
    if abs(value) > 2.0:
        return value / 100.0
    return value


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    output: Path
    cache: Path

    @classmethod
    def from_config(cls, config_path: Path, config: Mapping[str, Any]) -> "ProjectPaths":
        root = config_path.resolve().parent
        project = config.get("project", {})
        return cls(
            root=root,
            data=root / str(project.get("data_dir", "data")),
            output=root / str(project.get("output_dir", "outputs")),
            cache=root / str(project.get("cache_dir", "cache")),
        )

    def ensure(self) -> None:
        for path in (self.data, self.output, self.cache):
            path.mkdir(parents=True, exist_ok=True)


class SECClient:
    """Small, cached SEC client that observes the SEC fair-access limit."""

    DATA_BASE = "https://data.sec.gov"
    WWW_BASE = "https://www.sec.gov"

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_requests_per_second: float = 4.0,
        cache_hours: float = 12.0,
        timeout_seconds: int = 45,
        retries: int = 4,
    ) -> None:
        if not user_agent or "replace" in user_agent.lower() or "example.com" in user_agent.lower():
            raise ValueError(
                "SEC requires a descriptive User-Agent. Pass --user-agent \"Your Name your@email.com\" "
                "or set SEC_USER_AGENT."
            )
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.minimum_interval = 1.0 / max(0.5, min(max_requests_per_second, 9.0))
        self.cache_seconds = cache_hours * 3600.0
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html,application/xhtml+xml",
            }
        )

    def _cache_path(self, url: str, extension: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{extension}"

    def _is_fresh(self, path: Path) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) <= self.cache_seconds

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)

    def _request(self, url: str) -> requests.Response:
        parsed = urlparse(url)
        if parsed.hostname not in {"www.sec.gov", "data.sec.gov", "sec.gov"}:
            raise ValueError(f"SECClient refuses non-SEC URL: {url}")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                self._last_request = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait_seconds = min(30.0, 1.5 * (2**attempt))
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise RuntimeError(f"Failed SEC request after retries: {url}: {last_error}")

    def get_json(self, url: str) -> dict[str, Any]:
        path = self._cache_path(url, "json")
        if self._is_fresh(path):
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        response = self._request(url)
        data = response.json()
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp.replace(path)
        return data

    def get_text(self, url: str) -> str:
        path = self._cache_path(url, "html")
        if self._is_fresh(path):
            return path.read_text(encoding="utf-8", errors="replace")
        response = self._request(url)
        text = response.text
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return text

    def company_facts(self, cik: int | str) -> dict[str, Any]:
        cik_padded = str(int(cik)).zfill(10)
        return self.get_json(f"{self.DATA_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json")

    def submissions(self, cik: int | str) -> dict[str, Any]:
        cik_padded = str(int(cik)).zfill(10)
        return self.get_json(f"{self.DATA_BASE}/submissions/CIK{cik_padded}.json")

    def filing_html(self, cik: int | str, accession: str, primary_document: str) -> tuple[str, str]:
        accession_no_dash = accession.replace("-", "")
        url = (
            f"{self.WWW_BASE}/Archives/edgar/data/{int(cik)}/"
            f"{accession_no_dash}/{primary_document}"
        )
        return self.get_text(url), url


class CompanyFactsParser:
    """Convert SEC Company Facts into investor-friendly quarterly time series."""

    def __init__(self, history_years: int = 7) -> None:
        self.history_years = history_years
        self.cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=history_years + 2)

    @staticmethod
    def _choose_unit(units: Mapping[str, Any], preferred: str = "USD") -> tuple[str, Sequence[Mapping[str, Any]]]:
        if preferred in units:
            return preferred, units[preferred]
        for name, records in units.items():
            if name.upper().startswith(preferred.upper()):
                return name, records
        return "", []

    def _collect_records(
        self,
        companyfacts: Mapping[str, Any],
        concepts: Sequence[str],
        require_start: bool,
    ) -> pd.DataFrame:
        gaap = companyfacts.get("facts", {}).get("us-gaap", {})
        rows: list[dict[str, Any]] = []

        for alias_rank, concept in enumerate(concepts):
            concept_data = gaap.get(concept)
            if not isinstance(concept_data, Mapping):
                continue
            unit_name, records = self._choose_unit(concept_data.get("units", {}), "USD")
            for item in records:
                form = str(item.get("form", ""))
                if form not in ACCEPTED_FORMS:
                    continue
                start = item.get("start")
                end = item.get("end")
                if end is None or (require_start and start is None):
                    continue
                if not require_start and start is not None:
                    continue
                try:
                    value = float(item.get("val"))
                except (TypeError, ValueError):
                    continue
                row = {
                    "concept": concept,
                    "alias_rank": alias_rank,
                    "unit": unit_name,
                    "value": value,
                    "start": pd.to_datetime(start, errors="coerce") if start is not None else pd.NaT,
                    "end": pd.to_datetime(end, errors="coerce"),
                    "filed": pd.to_datetime(item.get("filed"), errors="coerce"),
                    "form": form,
                    "accn": item.get("accn", ""),
                    "frame": item.get("frame", ""),
                    "fy": item.get("fy"),
                    "fp": item.get("fp"),
                }
                if pd.isna(row["end"]):
                    continue
                rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "concept",
                    "alias_rank",
                    "unit",
                    "value",
                    "start",
                    "end",
                    "filed",
                    "form",
                    "accn",
                    "frame",
                    "fy",
                    "fp",
                ]
            )

        df = pd.DataFrame(rows)
        df = df[df["end"] >= self.cutoff].copy()
        # Later filings can restate prior periods. Keep the latest filed value for
        # each concept and exact period.
        keys = ["concept", "start", "end"] if require_start else ["concept", "end"]
        df = (
            df.sort_values(keys + ["filed", "accn"], na_position="first")
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )
        return df

    @staticmethod
    def _derive_quarters(records: pd.DataFrame, metric: str) -> pd.DataFrame:
        if records.empty:
            return pd.DataFrame(
                columns=["period_end", metric, f"{metric}_source_tag", f"{metric}_method", f"{metric}_filed"]
            )

        df = records.copy()
        df["duration_days"] = (df["end"] - df["start"]).dt.days + 1
        df = df[(df["duration_days"] >= 45) & (df["duration_days"] <= 410)].copy()
        candidates: list[dict[str, Any]] = []

        # Direct quarter disclosures are preferred.
        direct = df[df["duration_days"].between(45, 135)]
        for row in direct.itertuples(index=False):
            candidates.append(
                {
                    "period_end": row.end,
                    metric: row.value,
                    f"{metric}_source_tag": row.concept,
                    f"{metric}_method": "direct_quarter",
                    f"{metric}_filed": row.filed,
                    "method_rank": 0,
                    "alias_rank": row.alias_rank,
                }
            )

        # Cash-flow facts and some income-statement facts appear only as YTD.
        # Differencing observations with the same fiscal-year start produces the
        # missing standalone quarter.
        for _, group in df.groupby(["concept", "start"], dropna=False):
            group = group.sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
            previous: Any = None
            for row in group.itertuples(index=False):
                if previous is None:
                    previous = row
                    continue
                interval_days = (row.end - previous.end).days
                if 45 <= interval_days <= 140:
                    value = row.value - previous.value
                    candidates.append(
                        {
                            "period_end": row.end,
                            metric: value,
                            f"{metric}_source_tag": row.concept,
                            f"{metric}_method": "derived_from_ytd",
                            f"{metric}_filed": row.filed,
                            "method_rank": 1,
                            "alias_rank": row.alias_rank,
                        }
                    )
                previous = row

        if not candidates:
            return pd.DataFrame(
                columns=["period_end", metric, f"{metric}_source_tag", f"{metric}_method", f"{metric}_filed"]
            )

        result = pd.DataFrame(candidates)
        result = result.sort_values(
            ["period_end", "method_rank", "alias_rank", f"{metric}_filed"],
            ascending=[True, True, True, False],
            na_position="last",
        )
        result = result.drop_duplicates("period_end", keep="first")
        result = result.drop(columns=["method_rank", "alias_rank"])
        if metric == "capex":
            result[metric] = result[metric].abs()
        return result.reset_index(drop=True)

    @staticmethod
    def _instant_series(records: pd.DataFrame, metric: str) -> pd.DataFrame:
        if records.empty:
            return pd.DataFrame(
                columns=["period_end", metric, f"{metric}_source_tag", f"{metric}_method", f"{metric}_filed"]
            )
        df = records.sort_values(
            ["end", "alias_rank", "filed"],
            ascending=[True, True, False],
            na_position="last",
        ).copy()
        df = df.drop_duplicates("end", keep="first")
        return pd.DataFrame(
            {
                "period_end": df["end"],
                metric: df["value"],
                f"{metric}_source_tag": df["concept"],
                f"{metric}_method": "reported_instant",
                f"{metric}_filed": df["filed"],
            }
        ).reset_index(drop=True)

    def parse_company(
        self,
        companyfacts: Mapping[str, Any],
        ticker: str,
        configured_name: str | None = None,
        layer: str | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for metric, concepts in FLOW_METRICS.items():
            records = self._collect_records(companyfacts, concepts, require_start=True)
            frames.append(self._derive_quarters(records, metric))
        for metric, concepts in INSTANT_METRICS.items():
            records = self._collect_records(companyfacts, concepts, require_start=False)
            frames.append(self._instant_series(records, metric))

        merged: pd.DataFrame | None = None
        for frame in frames:
            if frame.empty:
                continue
            merged = frame if merged is None else merged.merge(frame, on="period_end", how="outer")

        if merged is None or merged.empty:
            return pd.DataFrame()

        merged = merged.sort_values("period_end").reset_index(drop=True)
        merged.insert(0, "ticker", ticker.upper())
        merged.insert(1, "company", configured_name or str(companyfacts.get("entityName", ticker)))
        merged.insert(2, "layer", layer or "")
        merged["period_end"] = pd.to_datetime(merged["period_end"])
        merged = merged[merged["period_end"] >= (pd.Timestamp.today() - pd.DateOffset(years=self.history_years))]
        merged = self.add_derived_metrics(merged)
        return merged.reset_index(drop=True)

    @staticmethod
    def _rolling_ttm(series: pd.Series, dates: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        raw = values.rolling(4, min_periods=4).sum()
        valid = pd.Series(False, index=series.index)
        for i in range(3, len(series)):
            span = (dates.iloc[i] - dates.iloc[i - 3]).days
            gaps = dates.iloc[i - 2 : i + 1].reset_index(drop=True) - dates.iloc[i - 3 : i].reset_index(drop=True)
            gap_days = [x.days for x in gaps]
            valid.iloc[i] = 230 <= span <= 430 and all(45 <= d <= 150 for d in gap_days)
        return raw.where(valid)

    @classmethod
    def add_derived_metrics(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("period_end").reset_index(drop=True)
        flow_cols = [c for c in FLOW_METRICS if c in out.columns]
        for col in flow_cols:
            out[f"{col}_ttm"] = cls._rolling_ttm(out[col], out["period_end"])
            out[f"{col}_ttm_yoy"] = safe_div(out[f"{col}_ttm"], out[f"{col}_ttm"].shift(4)) - 1.0
            out[f"{col}_ttm_change"] = out[f"{col}_ttm"] - out[f"{col}_ttm"].shift(4)

        for col in INSTANT_METRICS:
            if col in out.columns:
                out[f"{col}_yoy"] = safe_div(out[col], out[col].shift(4)) - 1.0

        if {"gross_profit_ttm", "revenue_ttm"}.issubset(out.columns):
            out["gross_margin_ttm"] = safe_div(out["gross_profit_ttm"], out["revenue_ttm"])
        if {"operating_income_ttm", "revenue_ttm"}.issubset(out.columns):
            out["operating_margin_ttm"] = safe_div(out["operating_income_ttm"], out["revenue_ttm"])
        if {"capex_ttm", "revenue_ttm"}.issubset(out.columns):
            out["capex_to_revenue"] = safe_div(out["capex_ttm"], out["revenue_ttm"])
        if {"capex_ttm", "operating_cash_flow_ttm"}.issubset(out.columns):
            out["capex_to_operating_cash_flow"] = safe_div(out["capex_ttm"], out["operating_cash_flow_ttm"])
            out["free_cash_flow_proxy_ttm"] = out["operating_cash_flow_ttm"] - out["capex_ttm"]
        if {"gross_profit_ttm", "capex_ttm"}.issubset(out.columns):
            out["gross_profit_to_capex"] = safe_div(out["gross_profit_ttm"], out["capex_ttm"])
        if {"gross_profit_ttm_change", "capex_ttm_change"}.issubset(out.columns):
            out["incremental_gross_profit_per_incremental_capex"] = safe_div(
                out["gross_profit_ttm_change"], out["capex_ttm_change"].where(out["capex_ttm_change"] > 0)
            )
        if {"gross_profit_ttm_yoy", "capex_ttm_yoy"}.issubset(out.columns):
            out["gross_profit_growth_minus_capex_growth"] = (
                out["gross_profit_ttm_yoy"] - out["capex_ttm_yoy"]
            )
        if {"accounts_receivable", "revenue_ttm"}.issubset(out.columns):
            out["receivable_days"] = safe_div(out["accounts_receivable"], out["revenue_ttm"]) * 365.0
        if {"accounts_receivable_yoy", "revenue_ttm_yoy"}.issubset(out.columns):
            out["receivable_growth_minus_revenue_growth"] = (
                out["accounts_receivable_yoy"] - out["revenue_ttm_yoy"]
            )
        if {"revenue_ttm", "ppe_net"}.issubset(out.columns):
            avg_ppe = (out["ppe_net"] + out["ppe_net"].shift(4)) / 2.0
            out["ppe_turnover"] = safe_div(out["revenue_ttm"], avg_ppe)
        if {"share_based_compensation_ttm", "revenue_ttm"}.issubset(out.columns):
            out["sbc_to_revenue"] = safe_div(out["share_based_compensation_ttm"], out["revenue_ttm"])
        if {"research_and_development_ttm", "revenue_ttm"}.issubset(out.columns):
            out["rd_to_revenue"] = safe_div(out["research_and_development_ttm"], out["revenue_ttm"])
        return out


def create_templates(paths: ProjectPaths) -> None:
    manual_path = paths.data / "manual_ai_metrics.csv"
    assumptions_path = paths.data / "cohort_assumptions.csv"
    if not manual_path.exists():
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(manual_path, index=False)
    if not assumptions_path.exists():
        pd.DataFrame(columns=ASSUMPTION_COLUMNS).to_csv(assumptions_path, index=False)


def load_manual_metrics(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    df = pd.read_csv(path)
    for col in MANUAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[MANUAL_COLUMNS].copy()
    if df.empty:
        return df
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce")
    for col in ["value", "low", "high", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["metric"] = df["metric"].astype(str).str.lower().str.strip()
    df["period_type"] = df["period_type"].fillna("point_in_time").astype(str).str.lower().str.strip()
    df["confidence"] = df["confidence"].fillna(0.5).clip(0.0, 1.0)
    df = df.dropna(subset=["as_of_date", "ticker", "metric", "value"])
    return df.sort_values(["ticker", "metric", "as_of_date"]).reset_index(drop=True)


def load_assumptions(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=ASSUMPTION_COLUMNS)
    df = pd.read_csv(path)
    for col in ASSUMPTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[ASSUMPTION_COLUMNS].copy()
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["scenario"] = df["scenario"].astype(str).str.lower().str.strip()
    for col in ASSUMPTION_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def scenario_value(row: pd.Series, scenario: str, metric: str) -> float:
    base = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
    low = pd.to_numeric(pd.Series([row.get("low")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("high")]), errors="coerce").iloc[0]
    low = base if pd.isna(low) else low
    high = base if pd.isna(high) else high
    if scenario == "base":
        return float(base)
    beneficial = metric in HIGH_IS_GOOD
    if scenario == "downside":
        return float(low if beneficial else high)
    return float(high if beneficial else low)


def asof_manual_row(
    manual: pd.DataFrame,
    ticker: str,
    metric: str,
    as_of_date: pd.Timestamp,
) -> pd.Series | None:
    if manual.empty:
        return None
    matches = manual[
        (manual["ticker"] == ticker.upper())
        & (manual["metric"] == metric.lower())
        & (manual["as_of_date"] <= as_of_date)
    ]
    if matches.empty:
        return None
    return matches.sort_values("as_of_date").iloc[-1]


def manual_growth(
    manual: pd.DataFrame,
    ticker: str,
    metric: str,
    scenario: str,
    as_of_date: pd.Timestamp,
    lookback_days: int = 365,
) -> float:
    rows = manual[
        (manual["ticker"] == ticker.upper())
        & (manual["metric"] == metric.lower())
        & (manual["as_of_date"] <= as_of_date)
    ].sort_values("as_of_date")
    if len(rows) < 2:
        return np.nan
    current = rows.iloc[-1]
    earlier = rows.iloc[:-1].copy()
    if earlier.empty:
        return np.nan
    earlier["days_ago"] = (pd.Timestamp(current["as_of_date"]) - earlier["as_of_date"]).dt.days
    tolerance = 90
    prior_candidates = earlier[
        earlier["days_ago"].between(lookback_days - tolerance, lookback_days + tolerance)
    ].copy()
    if prior_candidates.empty:
        return np.nan
    prior_candidates["distance_to_target"] = (prior_candidates["days_ago"] - lookback_days).abs()
    prior = prior_candidates.sort_values(["distance_to_target", "as_of_date"]).iloc[0]
    current_value = scenario_value(current, scenario, metric)
    prior_value = scenario_value(prior, scenario, metric)
    return safe_div(current_value, prior_value) - 1.0



def piecewise_score(value: float, points: Sequence[tuple[float, float]]) -> float:
    """Transparent linear score bounded by the supplied threshold points."""
    if pd.isna(value):
        return np.nan
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    return float(np.interp(float(value), xs, ys, left=ys[0], right=ys[-1]))


def entity_lookup_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    entries = list(config.get("universe", [])) + list(config.get("manual_entities", []))
    for item in entries:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        lookup[ticker] = {
            "company": str(item.get("name", ticker)),
            "layer": str(item.get("layer", "")),
        }
    return lookup


def build_paid_useful_work_scores(
    manual: pd.DataFrame,
    entity_lookup: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Create a public-data proxy score for paid, persistent, successful AI work.

    This is a score rather than a claimed count of economically useful tokens.
    It only uses components actually supplied by the investor and reports data
    completeness separately.
    """
    if manual.empty:
        return pd.DataFrame()

    weights = {
        "paid_demand": 0.25,
        "production": 0.20,
        "success": 0.15,
        "persistence": 0.20,
        "customer_value": 0.10,
        "margin": 0.10,
    }
    total_weight = sum(weights.values())
    rows: list[dict[str, Any]] = []

    def latest_row(ticker: str, metric: str, as_of: pd.Timestamp) -> pd.Series | None:
        return asof_manual_row(manual, ticker, metric, as_of)

    for ticker, ticker_rows in manual.groupby("ticker"):
        as_of = pd.Timestamp(ticker_rows["as_of_date"].max())
        entity = entity_lookup.get(ticker, {})
        manual_company = ticker_rows.sort_values("as_of_date")["company"].iloc[-1]
        if pd.isna(manual_company) or not str(manual_company).strip():
            manual_company = ticker
        company = str(entity.get("company") or manual_company)
        layer = str(entity.get("layer", ""))

        for scenario in SCENARIOS:
            component_scores: dict[str, float] = {}
            component_confidences: dict[str, float] = {}
            component_values: dict[str, float] = {}
            source_metrics: dict[str, str] = {}

            demand_priorities = [
                "paid_useful_work_index",
                "ai_revenue",
                "ai_revenue_run_rate",
                "ai_arr",
                "ai_rpo",
                "paid_enterprise_customers",
                "paid_token_index",
            ]
            for metric in demand_priorities:
                growth = manual_growth(manual, ticker, metric, scenario, as_of)
                if pd.notna(growth):
                    row = latest_row(ticker, metric, as_of)
                    score = piecewise_score(
                        growth,
                        [(-0.20, 0), (0.00, 30), (0.20, 60), (0.50, 85), (1.00, 100)],
                    )
                    # Token growth is a weaker proxy than revenue or explicit
                    # useful-work growth, so cap its component score.
                    if metric == "paid_token_index":
                        score = min(score, 75.0)
                    component_scores["paid_demand"] = score
                    component_values["paid_demand"] = growth
                    component_confidences["paid_demand"] = float(row.get("confidence", 0.5)) if row is not None else 0.5
                    source_metrics["paid_demand"] = metric
                    break

            production = latest_row(ticker, "production_share", as_of)
            if production is not None:
                value = normalize_ratio(scenario_value(production, scenario, "production_share"))
                component_scores["production"] = piecewise_score(
                    value, [(0.20, 0), (0.40, 30), (0.60, 60), (0.80, 85), (0.95, 100)]
                )
                component_values["production"] = value
                component_confidences["production"] = float(production.get("confidence", 0.5))
                source_metrics["production"] = "production_share"

            success = latest_row(ticker, "task_success_rate", as_of)
            if success is not None:
                value = normalize_ratio(scenario_value(success, scenario, "task_success_rate"))
                component_scores["success"] = piecewise_score(
                    value, [(0.30, 0), (0.50, 30), (0.70, 60), (0.85, 85), (0.95, 100)]
                )
                component_values["success"] = value
                component_confidences["success"] = float(success.get("confidence", 0.5))
                source_metrics["success"] = "task_success_rate"

            nrr = latest_row(ticker, "net_revenue_retention", as_of)
            renewal = latest_row(ticker, "renewal_rate", as_of)
            if nrr is not None:
                value = normalize_ratio(scenario_value(nrr, scenario, "net_revenue_retention"))
                component_scores["persistence"] = piecewise_score(
                    value, [(0.70, 0), (0.90, 35), (1.00, 65), (1.20, 100)]
                )
                component_values["persistence"] = value
                component_confidences["persistence"] = float(nrr.get("confidence", 0.5))
                source_metrics["persistence"] = "net_revenue_retention"
            elif renewal is not None:
                value = normalize_ratio(scenario_value(renewal, scenario, "renewal_rate"))
                component_scores["persistence"] = piecewise_score(
                    value, [(0.50, 0), (0.75, 40), (0.90, 75), (1.00, 100)]
                )
                component_values["persistence"] = value
                component_confidences["persistence"] = float(renewal.get("confidence", 0.5))
                source_metrics["persistence"] = "renewal_rate"

            value_growth = manual_growth(
                manual, ticker, "customer_value_index", scenario, as_of
            )
            value_row = latest_row(ticker, "customer_value_index", as_of)
            if pd.notna(value_growth) and value_row is not None:
                component_scores["customer_value"] = piecewise_score(
                    value_growth,
                    [(-0.20, 0), (0.00, 35), (0.15, 60), (0.35, 85), (0.70, 100)],
                )
                component_values["customer_value"] = value_growth
                component_confidences["customer_value"] = float(value_row.get("confidence", 0.5))
                source_metrics["customer_value"] = "customer_value_index"

            margin = latest_row(ticker, "ai_contribution_margin", as_of)
            if margin is not None:
                value = normalize_ratio(scenario_value(margin, scenario, "ai_contribution_margin"))
                component_scores["margin"] = piecewise_score(
                    value, [(0.00, 0), (0.10, 25), (0.25, 50), (0.40, 75), (0.60, 100)]
                )
                component_values["margin"] = value
                component_confidences["margin"] = float(margin.get("confidence", 0.5))
                source_metrics["margin"] = "ai_contribution_margin"

            available_weight = sum(weights[name] for name in component_scores)
            completeness = available_weight / total_weight
            if available_weight >= 0.35:
                score = sum(component_scores[name] * weights[name] for name in component_scores) / available_weight
                confidence = sum(
                    component_confidences[name] * weights[name] for name in component_scores
                ) / available_weight
            else:
                score = np.nan
                confidence = np.nan

            if pd.isna(score):
                status = "INSUFFICIENT_DATA"
            elif score >= 70:
                status = "STRONG"
            elif score >= 50:
                status = "MIXED"
            else:
                status = "WEAK"

            output: dict[str, Any] = {
                "ticker": ticker,
                "company": company,
                "layer": layer,
                "as_of_date": as_of,
                "scenario": scenario,
                "paid_useful_work_score": score,
                "puw_status": status,
                "puw_data_completeness": completeness,
                "puw_confidence": confidence,
                "available_component_weight": available_weight,
            }
            for name in weights:
                output[f"{name}_score"] = component_scores.get(name, np.nan)
                output[f"{name}_value"] = component_values.get(name, np.nan)
                output[f"{name}_source_metric"] = source_metrics.get(name, "")
            rows.append(output)

    return pd.DataFrame(rows)


def build_ecosystem_summary(
    sec_quarters: pd.DataFrame,
    coverage: pd.DataFrame,
    puw_scores: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate without summing revenue across multiple value-chain layers."""
    analysis = config.get("analysis", {})
    capital_owner_layers = set(analysis.get("capital_owner_layers", []))
    downstream_layers = set(analysis.get("downstream_signal_layers", []))

    upstream_capex_growth = np.nan
    upstream_capex_ttm = np.nan
    if not sec_quarters.empty:
        latest_sec = (
            sec_quarters.sort_values("period_end")
            .groupby("ticker", as_index=False)
            .tail(1)
        )
        owners = latest_sec[latest_sec["layer"].isin(capital_owner_layers)].copy()
        valid = owners.dropna(subset=["capex_ttm", "capex_ttm_yoy"])
        if not valid.empty:
            current = pd.to_numeric(valid["capex_ttm"], errors="coerce").sum()
            prior = (
                pd.to_numeric(valid["capex_ttm"], errors="coerce")
                / (1.0 + pd.to_numeric(valid["capex_ttm_yoy"], errors="coerce"))
            ).replace([np.inf, -np.inf], np.nan).sum(min_count=1)
            upstream_capex_ttm = current
            upstream_capex_growth = safe_div(current, prior) - 1.0

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        total_profit = np.nan
        total_charge = np.nan
        ecosystem_coverage = np.nan
        coverage_confidence = np.nan
        coverage_company_count = 0
        latest_period = pd.NaT

        if not coverage.empty:
            scenario_cov = coverage[coverage["scenario"] == scenario]
            latest_cov = (
                scenario_cov.sort_values("period_end")
                .groupby("ticker", as_index=False)
                .tail(1)
            )
            latest_cov = latest_cov[latest_cov["layer"].isin(capital_owner_layers)]
            latest_cov = latest_cov.dropna(
                subset=["ai_contribution_profit_coverage_numerator", "ai_total_capital_charge_ttm"]
            )
            latest_cov = latest_cov[latest_cov["ai_total_capital_charge_ttm"] > 0]
            if not latest_cov.empty:
                total_profit = latest_cov["ai_contribution_profit_coverage_numerator"].sum()
                total_charge = latest_cov["ai_total_capital_charge_ttm"].sum()
                ecosystem_coverage = safe_div(total_profit, total_charge)
                weights = latest_cov["ai_total_capital_charge_ttm"].clip(lower=0)
                if weights.sum() > 0:
                    coverage_confidence = np.average(
                        latest_cov["ai_input_confidence"].fillna(0), weights=weights
                    )
                coverage_company_count = len(latest_cov)
                latest_period = latest_cov["period_end"].max()

        downstream_score = np.nan
        downstream_confidence = np.nan
        downstream_completeness = np.nan
        downstream_company_count = 0
        if not puw_scores.empty:
            scenario_puw = puw_scores[
                (puw_scores["scenario"] == scenario)
                & (puw_scores["layer"].isin(downstream_layers))
            ].dropna(subset=["paid_useful_work_score"])
            if not scenario_puw.empty:
                weights = (
                    scenario_puw["puw_data_completeness"].fillna(0)
                    * scenario_puw["puw_confidence"].fillna(0)
                )
                if weights.sum() > 0:
                    downstream_score = np.average(
                        scenario_puw["paid_useful_work_score"], weights=weights
                    )
                    downstream_confidence = np.average(
                        scenario_puw["puw_confidence"], weights=weights
                    )
                    downstream_completeness = np.average(
                        scenario_puw["puw_data_completeness"], weights=weights
                    )
                downstream_company_count = len(scenario_puw)

        if pd.isna(ecosystem_coverage) or pd.isna(downstream_score):
            signal = "INSUFFICIENT_DATA"
        elif ecosystem_coverage >= 1.25 and downstream_score >= 65:
            signal = "HEALTHY_CATCHUP"
        elif (
            ecosystem_coverage < 0.80 or downstream_score < 45
        ) and pd.notna(upstream_capex_growth) and upstream_capex_growth > 0.15:
            signal = "OVERBUILD_WARNING"
        else:
            signal = "MIXED"

        rows.append(
            {
                "scenario": scenario,
                "latest_period_end": latest_period,
                "capital_owner_ai_contribution_profit_annualized": total_profit,
                "capital_owner_ai_capital_charge_ttm": total_charge,
                "capital_owner_ai_economic_coverage": ecosystem_coverage,
                "capital_owner_coverage_confidence": coverage_confidence,
                "capital_owner_company_count": coverage_company_count,
                "downstream_paid_useful_work_score": downstream_score,
                "downstream_puw_confidence": downstream_confidence,
                "downstream_puw_completeness": downstream_completeness,
                "downstream_company_count": downstream_company_count,
                "capital_owner_total_capex_ttm": upstream_capex_ttm,
                "capital_owner_total_capex_growth_yoy": upstream_capex_growth,
                "ecosystem_signal": signal,
            }
        )
    return pd.DataFrame(rows)


def assumption_value(
    assumptions: pd.DataFrame,
    ticker: str,
    scenario: str,
    column: str,
    default: float,
) -> float:
    if assumptions.empty:
        return float(default)
    rows = assumptions[
        (assumptions["ticker"].isin([ticker.upper(), "DEFAULT"]))
        & (assumptions["scenario"] == scenario)
    ].copy()
    if rows.empty:
        return float(default)
    rows["ticker_rank"] = (rows["ticker"] != ticker.upper()).astype(int)
    row = rows.sort_values("ticker_rank").iloc[0]
    value = row.get(column)
    return float(value) if pd.notna(value) else float(default)


def estimate_ai_quarter_inputs(
    company_quarters: pd.DataFrame,
    manual: pd.DataFrame,
    ticker: str,
    scenario: str,
) -> pd.DataFrame:
    """Attach scenario-specific AI capex and contribution-profit estimates."""
    df = company_quarters.sort_values("period_end").copy()
    ai_capex_values: list[float] = []
    ai_profit_values: list[float] = []
    ai_revenue_values: list[float] = []
    confidence_values: list[float] = []
    input_descriptions: list[str] = []

    for row in df.itertuples(index=False):
        date = pd.Timestamp(row.period_end)
        used_confidence: list[float] = []
        descriptions: list[str] = []

        capex_direct = asof_manual_row(manual, ticker, "ai_capex_dollars", date)
        capex_share = asof_manual_row(manual, ticker, "ai_capex_share", date)
        total_capex = getattr(row, "capex", np.nan)
        if capex_direct is not None:
            ai_capex = quarterly_flow(
                scenario_value(capex_direct, scenario, "ai_capex_dollars"),
                str(capex_direct.get("period_type", "quarter")),
            )
            used_confidence.append(float(capex_direct.get("confidence", 0.5)))
            descriptions.append("manual_ai_capex")
        elif capex_share is not None and pd.notna(total_capex):
            share = normalize_ratio(scenario_value(capex_share, scenario, "ai_capex_share"))
            ai_capex = float(total_capex) * share
            used_confidence.append(float(capex_share.get("confidence", 0.5)))
            descriptions.append("sec_capex_x_manual_share")
        else:
            ai_capex = np.nan

        profit_direct = asof_manual_row(manual, ticker, "ai_contribution_profit", date)
        profit_run_rate = asof_manual_row(manual, ticker, "ai_contribution_profit_run_rate", date)
        revenue_direct = asof_manual_row(manual, ticker, "ai_revenue", date)
        revenue_run_rate = asof_manual_row(manual, ticker, "ai_revenue_run_rate", date)
        margin = asof_manual_row(manual, ticker, "ai_contribution_margin", date)
        attribution = asof_manual_row(manual, ticker, "ai_incremental_gp_attribution_share", date)

        ai_revenue = np.nan
        if revenue_direct is not None:
            ai_revenue = quarterly_flow(
                scenario_value(revenue_direct, scenario, "ai_revenue"),
                str(revenue_direct.get("period_type", "quarter")),
            )
            used_confidence.append(float(revenue_direct.get("confidence", 0.5)))
        elif revenue_run_rate is not None:
            ai_revenue = scenario_value(revenue_run_rate, scenario, "ai_revenue_run_rate") / 4.0
            used_confidence.append(float(revenue_run_rate.get("confidence", 0.5)))

        if profit_direct is not None:
            ai_profit = quarterly_flow(
                scenario_value(profit_direct, scenario, "ai_contribution_profit"),
                str(profit_direct.get("period_type", "quarter")),
            )
            used_confidence.append(float(profit_direct.get("confidence", 0.5)))
            descriptions.append("manual_ai_contribution_profit")
        elif profit_run_rate is not None:
            ai_profit = scenario_value(profit_run_rate, scenario, "ai_contribution_profit_run_rate") / 4.0
            used_confidence.append(float(profit_run_rate.get("confidence", 0.5)))
            descriptions.append("manual_ai_profit_run_rate")
        elif pd.notna(ai_revenue) and margin is not None:
            margin_value = normalize_ratio(scenario_value(margin, scenario, "ai_contribution_margin"))
            ai_profit = ai_revenue * margin_value
            used_confidence.append(float(margin.get("confidence", 0.5)))
            descriptions.append("ai_revenue_x_manual_margin")
        elif attribution is not None:
            gp_change_ttm = getattr(row, "gross_profit_ttm_change", np.nan)
            if pd.notna(gp_change_ttm):
                share = normalize_ratio(
                    scenario_value(attribution, scenario, "ai_incremental_gp_attribution_share")
                )
                ai_profit = max(0.0, float(gp_change_ttm) * share / 4.0)
                used_confidence.append(float(attribution.get("confidence", 0.5)))
                descriptions.append("incremental_gp_x_ai_attribution")
            else:
                ai_profit = np.nan
        else:
            ai_profit = np.nan

        ai_capex_values.append(ai_capex)
        ai_profit_values.append(ai_profit)
        ai_revenue_values.append(ai_revenue)
        confidence_values.append(float(np.mean(used_confidence)) if used_confidence else np.nan)
        input_descriptions.append(";".join(sorted(set(descriptions))))

    df["ai_capex_estimate"] = ai_capex_values
    df["ai_contribution_profit_estimate"] = ai_profit_values
    df["ai_revenue_estimate"] = ai_revenue_values
    df["ai_contribution_profit_annualized"] = df["ai_contribution_profit_estimate"] * 4.0
    df["ai_revenue_annualized"] = df["ai_revenue_estimate"] * 4.0
    df["ai_input_confidence"] = confidence_values
    df["ai_input_method"] = input_descriptions
    return df


def build_capital_stock(
    inputs: pd.DataFrame,
    useful_life_years: float,
    hurdle_rate: float,
    maintenance_capex_pct: float,
) -> pd.DataFrame:
    """Straight-line vintage model using quarterly placed-in-service cohorts."""
    df = inputs.sort_values("period_end").reset_index(drop=True).copy()
    life_quarters = max(4, int(round(useful_life_years * 4.0)))
    vintages: list[tuple[int, float]] = []
    opening_net = 0.0
    gross_active_values: list[float] = []
    net_values: list[float] = []
    depreciation_values: list[float] = []
    required_return_values: list[float] = []
    maintenance_values: list[float] = []
    capital_charge_values: list[float] = []

    for quarter_index, row in df.iterrows():
        capex = row.get("ai_capex_estimate", np.nan)
        capex = 0.0 if pd.isna(capex) else max(0.0, float(capex))
        if capex > 0:
            vintages.append((quarter_index, capex))

        depreciation = 0.0
        gross_active = 0.0
        closing_net = 0.0
        active_vintages: list[tuple[int, float]] = []
        for start_index, amount in vintages:
            age = quarter_index - start_index
            if age >= life_quarters:
                continue
            active_vintages.append((start_index, amount))
            gross_active += amount
            depreciation += amount / life_quarters
            remaining_fraction = max(0.0, 1.0 - (age + 1.0) / life_quarters)
            closing_net += amount * remaining_fraction
        vintages = active_vintages

        average_net = (opening_net + capex + closing_net) / 2.0
        required_return = average_net * hurdle_rate / 4.0
        maintenance = gross_active * maintenance_capex_pct / 4.0
        capital_charge = depreciation + required_return + maintenance

        gross_active_values.append(gross_active)
        net_values.append(closing_net)
        depreciation_values.append(depreciation)
        required_return_values.append(required_return)
        maintenance_values.append(maintenance)
        capital_charge_values.append(capital_charge)
        opening_net = closing_net

    df["ai_gross_capital_stock"] = gross_active_values
    df["ai_net_capital_stock"] = net_values
    df["ai_economic_depreciation"] = depreciation_values
    df["ai_required_return_charge"] = required_return_values
    df["ai_maintenance_charge"] = maintenance_values
    df["ai_total_capital_charge"] = capital_charge_values

    for col in [
        "ai_capex_estimate",
        "ai_contribution_profit_estimate",
        "ai_revenue_estimate",
        "ai_economic_depreciation",
        "ai_required_return_charge",
        "ai_maintenance_charge",
        "ai_total_capital_charge",
    ]:
        df[f"{col}_ttm"] = df[col].rolling(4, min_periods=4).sum()

    df["ai_contribution_profit_coverage_numerator"] = (
        df["ai_contribution_profit_estimate_ttm"]
        .combine_first(df["ai_contribution_profit_annualized"])
    )
    df["ai_revenue_coverage_run_rate"] = (
        df["ai_revenue_estimate_ttm"].combine_first(df["ai_revenue_annualized"])
    )
    df["ai_economic_coverage_ttm"] = safe_div(
        df["ai_contribution_profit_estimate_ttm"], df["ai_total_capital_charge_ttm"]
    )
    df["ai_economic_coverage_run_rate"] = safe_div(
        df["ai_contribution_profit_annualized"], df["ai_total_capital_charge_ttm"]
    )
    df["ai_economic_coverage"] = df["ai_economic_coverage_ttm"].combine_first(
        df["ai_economic_coverage_run_rate"]
    )
    df["ai_coverage_basis"] = np.select(
        [
            df["ai_economic_coverage_ttm"].notna(),
            df["ai_economic_coverage_run_rate"].notna(),
        ],
        ["ttm", "annualized_current_run_rate"],
        default="unavailable",
    )
    df["ai_contribution_return_on_net_capital"] = safe_div(
        df["ai_contribution_profit_coverage_numerator"], df["ai_net_capital_stock"]
    )
    df["ai_reinvestment_coverage"] = safe_div(
        df["ai_contribution_profit_coverage_numerator"], df["ai_capex_estimate_ttm"]
    )
    df["ai_coverage_change_yoy"] = df["ai_economic_coverage"] - df["ai_economic_coverage"].shift(4)
    return df


def classify_coverage(coverage: float, trend: float, confidence: float, min_confidence: float) -> str:
    if pd.isna(coverage) or pd.isna(confidence) or confidence < min_confidence:
        return "INSUFFICIENT_AI_DISCLOSURE"
    if coverage >= 1.25 and (pd.isna(trend) or trend >= -0.10):
        return "SUPPORTED"
    if coverage < 0.80 and (pd.isna(trend) or trend <= 0.0):
        return "NOT_SUPPORTED"
    return "WATCH"


def build_ai_coverage(
    sec_quarters: pd.DataFrame,
    manual: pd.DataFrame,
    assumptions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if sec_quarters.empty:
        return pd.DataFrame()
    analysis = config.get("analysis", {})
    min_confidence = float(analysis.get("minimum_manual_confidence", 0.25))
    default_hurdle = float(analysis.get("hurdle_rate", 0.15))
    default_lives = analysis.get("default_ai_asset_life_years", {})
    default_maint = analysis.get("default_maintenance_capex_pct", {})
    frames: list[pd.DataFrame] = []

    for ticker, company_df in sec_quarters.groupby("ticker"):
        for scenario in SCENARIOS:
            life = assumption_value(
                assumptions,
                ticker,
                scenario,
                "ai_asset_life_years",
                float(default_lives.get(scenario, 4.0)),
            )
            hurdle = assumption_value(
                assumptions, ticker, scenario, "hurdle_rate", default_hurdle
            )
            maintenance = assumption_value(
                assumptions,
                ticker,
                scenario,
                "maintenance_capex_pct",
                float(default_maint.get(scenario, 0.03)),
            )
            inputs = estimate_ai_quarter_inputs(company_df, manual, ticker, scenario)
            modeled = build_capital_stock(inputs, life, hurdle, maintenance)
            modeled["scenario"] = scenario
            modeled["assumed_ai_asset_life_years"] = life
            modeled["assumed_hurdle_rate"] = hurdle
            modeled["assumed_maintenance_capex_pct"] = maintenance
            modeled["catchup_status"] = [
                classify_coverage(c, t, conf, min_confidence)
                for c, t, conf in zip(
                    modeled["ai_economic_coverage"],
                    modeled["ai_coverage_change_yoy"],
                    modeled["ai_input_confidence"],
                )
            ]
            frames.append(modeled)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _default_utilization(year: int, mature: float, y1: float, y2: float) -> float:
    if year == 1:
        return y1
    if year == 2:
        return y2
    return mature


def compute_cohort_irr(
    capex: float,
    revenue_per_capital: float,
    contribution_margin: float,
    useful_life_years: float,
    year1_utilization: float,
    year2_utilization: float,
    mature_utilization: float,
    annual_price_change: float,
    annual_efficiency_capture: float,
    maintenance_capex_pct: float,
    tax_rate: float,
    salvage_pct: float,
) -> tuple[float, list[float]]:
    if npf is None:
        return np.nan, []
    life = max(1, int(round(useful_life_years)))
    cashflows = [-float(capex)]
    for year in range(1, life + 1):
        utilization = _default_utilization(
            year, mature_utilization, year1_utilization, year2_utilization
        )
        net_price_efficiency = ((1.0 + annual_price_change) * (1.0 + annual_efficiency_capture)) ** (year - 1)
        revenue = capex * revenue_per_capital * utilization * net_price_efficiency
        contribution = revenue * contribution_margin
        maintenance = capex * maintenance_capex_pct
        after_tax_cash = (contribution - maintenance) * (1.0 - tax_rate)
        if year == life:
            after_tax_cash += capex * salvage_pct
        cashflows.append(after_tax_cash)
    try:
        irr = float(npf.irr(cashflows))
    except Exception:
        irr = np.nan
    return irr, cashflows


def build_cohort_irr_table(
    coverage: pd.DataFrame,
    assumptions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    analysis = config.get("analysis", {})
    default_tax = float(analysis.get("default_tax_rate", 0.21))
    rows: list[dict[str, Any]] = []

    latest = (
        coverage.sort_values("period_end")
        .groupby(["ticker", "scenario"], as_index=False)
        .tail(1)
    )
    for record in latest.to_dict("records"):
        ticker = record["ticker"]
        scenario = record["scenario"]
        capex = record.get("ai_capex_estimate_ttm")
        if pd.isna(capex) or float(capex) <= 0:
            continue
        observed_rpc = safe_div(
            record.get("ai_revenue_coverage_run_rate"), record.get("ai_gross_capital_stock")
        )
        revenue_per_capital = assumption_value(
            assumptions,
            ticker,
            scenario,
            "revenue_per_capital_at_full_utilization",
            float(observed_rpc) if pd.notna(observed_rpc) else np.nan,
        )
        contribution_margin = assumption_value(
            assumptions,
            ticker,
            scenario,
            "contribution_margin",
            safe_div(
                record.get("ai_contribution_profit_coverage_numerator"),
                record.get("ai_revenue_coverage_run_rate"),
            ),
        )
        if pd.isna(revenue_per_capital) or pd.isna(contribution_margin):
            continue
        life = float(record.get("assumed_ai_asset_life_years", 4.0))
        maintenance = float(record.get("assumed_maintenance_capex_pct", 0.03))
        hurdle = float(record.get("assumed_hurdle_rate", 0.15))
        y1 = assumption_value(assumptions, ticker, scenario, "year1_utilization", 0.55)
        y2 = assumption_value(assumptions, ticker, scenario, "year2_utilization", 0.75)
        mature = assumption_value(assumptions, ticker, scenario, "mature_utilization", 0.85)
        price_change = assumption_value(assumptions, ticker, scenario, "annual_price_change", -0.10)
        efficiency = assumption_value(
            assumptions, ticker, scenario, "annual_efficiency_capture", 0.10
        )
        tax = assumption_value(assumptions, ticker, scenario, "tax_rate", default_tax)
        salvage = assumption_value(assumptions, ticker, scenario, "salvage_pct", 0.05)

        irr, cashflows = compute_cohort_irr(
            capex=float(capex),
            revenue_per_capital=float(revenue_per_capital),
            contribution_margin=normalize_ratio(float(contribution_margin)),
            useful_life_years=life,
            year1_utilization=normalize_ratio(y1),
            year2_utilization=normalize_ratio(y2),
            mature_utilization=normalize_ratio(mature),
            annual_price_change=normalize_ratio(price_change),
            annual_efficiency_capture=normalize_ratio(efficiency),
            maintenance_capex_pct=normalize_ratio(maintenance),
            tax_rate=normalize_ratio(tax),
            salvage_pct=normalize_ratio(salvage),
        )
        if pd.isna(irr):
            status = "INSUFFICIENT_OR_NONCONVERGENT"
        elif irr >= hurdle:
            status = "ABOVE_HURDLE"
        elif irr >= max(0.08, hurdle * 0.65):
            status = "MARGINAL"
        else:
            status = "BELOW_HURDLE"
        rows.append(
            {
                "ticker": ticker,
                "company": record.get("company", ""),
                "layer": record.get("layer", ""),
                "period_end": record.get("period_end"),
                "scenario": scenario,
                "modeled_capex_cohort": float(capex),
                "revenue_per_capital_at_full_utilization": float(revenue_per_capital),
                "contribution_margin": normalize_ratio(float(contribution_margin)),
                "useful_life_years": life,
                "cohort_irr": irr,
                "hurdle_rate": hurdle,
                "irr_status": status,
                "cashflows": json.dumps(cashflows),
            }
        )
    return pd.DataFrame(rows)


def latest_manual_summary(manual: pd.DataFrame) -> pd.DataFrame:
    if manual.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (ticker, metric), group in manual.groupby(["ticker", "metric"]):
        latest = group.sort_values("as_of_date").iloc[-1]
        for scenario in SCENARIOS:
            rows.append(
                {
                    "ticker": ticker,
                    "company": latest.get("company", ""),
                    "metric": metric,
                    "scenario": scenario,
                    "as_of_date": latest["as_of_date"],
                    "value": scenario_value(latest, scenario, metric),
                    "unit": latest.get("unit", ""),
                    "period_type": latest.get("period_type", ""),
                    "confidence": latest.get("confidence", np.nan),
                    "source_name": latest.get("source_name", ""),
                    "source_url": latest.get("source_url", ""),
                    "source_date": latest.get("source_date", pd.NaT),
                    "reported_or_estimated": latest.get("reported_or_estimated", ""),
                    "notes": latest.get("notes", ""),
                    "yoy_growth": manual_growth(
                        manual, ticker, metric, scenario, pd.Timestamp(latest["as_of_date"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_public_scorecard(
    sec_quarters: pd.DataFrame,
    coverage: pd.DataFrame,
    cohort_irr: pd.DataFrame,
    manual: pd.DataFrame,
    puw_scores: pd.DataFrame,
    entity_lookup: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tickers = sorted(
        set(sec_quarters.get("ticker", pd.Series(dtype=str)).dropna())
        | set(manual.get("ticker", pd.Series(dtype=str)).dropna())
    )

    for ticker in tickers:
        sec_rows = sec_quarters[sec_quarters["ticker"] == ticker].sort_values("period_end")
        sec_latest = sec_rows.iloc[-1] if not sec_rows.empty else pd.Series(dtype=object)
        base_coverage = coverage[
            (coverage["ticker"] == ticker) & (coverage["scenario"] == "base")
        ].sort_values("period_end") if not coverage.empty else pd.DataFrame()
        cov_latest = base_coverage.iloc[-1] if not base_coverage.empty else pd.Series(dtype=object)
        base_irr = cohort_irr[
            (cohort_irr["ticker"] == ticker) & (cohort_irr["scenario"] == "base")
        ] if not cohort_irr.empty else pd.DataFrame()
        irr_latest = base_irr.iloc[-1] if not base_irr.empty else pd.Series(dtype=object)
        base_puw = puw_scores[
            (puw_scores["ticker"] == ticker) & (puw_scores["scenario"] == "base")
        ].sort_values("as_of_date") if not puw_scores.empty else pd.DataFrame()
        puw_latest = base_puw.iloc[-1] if not base_puw.empty else pd.Series(dtype=object)

        manual_ticker = manual[manual["ticker"] == ticker]
        manual_metrics = sorted(manual_ticker["metric"].unique().tolist()) if not manual_ticker.empty else []
        downstream_metric = None
        downstream_growth = np.nan
        for metric in [
            "paid_useful_work_index",
            "ai_revenue",
            "ai_revenue_run_rate",
            "ai_arr",
            "ai_rpo",
            "paid_enterprise_customers",
            "paid_token_index",
        ]:
            rows_m = manual_ticker[manual_ticker["metric"] == metric]
            if not rows_m.empty:
                downstream_metric = metric
                downstream_growth = manual_growth(
                    manual,
                    ticker,
                    metric,
                    "base",
                    pd.Timestamp(rows_m["as_of_date"].max()),
                )
                break

        manual_company = (
            manual_ticker.sort_values("as_of_date")["company"].iloc[-1]
            if not manual_ticker.empty
            else np.nan
        )
        manual_latest_date = (
            manual_ticker["as_of_date"].max() if not manual_ticker.empty else pd.NaT
        )

        entity = entity_lookup.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "company": first_non_null(
                    [
                        sec_latest.get("company"),
                        cov_latest.get("company"),
                        puw_latest.get("company"),
                        manual_company,
                        entity.get("company"),
                    ]
                ),
                "layer": first_non_null(
                    [sec_latest.get("layer"), cov_latest.get("layer"), puw_latest.get("layer"), entity.get("layer")]
                ),
                "period_end": first_non_null(
                    [sec_latest.get("period_end"), cov_latest.get("period_end"), manual_latest_date]
                ),
                "revenue_ttm": sec_latest.get("revenue_ttm", np.nan),
                "revenue_growth_yoy": sec_latest.get("revenue_ttm_yoy", np.nan),
                "gross_profit_ttm": sec_latest.get("gross_profit_ttm", np.nan),
                "gross_profit_growth_yoy": sec_latest.get("gross_profit_ttm_yoy", np.nan),
                "capex_ttm": sec_latest.get("capex_ttm", np.nan),
                "capex_growth_yoy": sec_latest.get("capex_ttm_yoy", np.nan),
                "gross_profit_growth_minus_capex_growth": sec_latest.get(
                    "gross_profit_growth_minus_capex_growth", np.nan
                ),
                "incremental_gp_per_incremental_capex": sec_latest.get(
                    "incremental_gross_profit_per_incremental_capex", np.nan
                ),
                "capex_to_operating_cash_flow": sec_latest.get(
                    "capex_to_operating_cash_flow", np.nan
                ),
                "receivable_growth_minus_revenue_growth": sec_latest.get(
                    "receivable_growth_minus_revenue_growth", np.nan
                ),
                "downstream_metric": downstream_metric,
                "downstream_metric_yoy_growth": downstream_growth,
                "paid_useful_work_score_base": puw_latest.get("paid_useful_work_score", np.nan),
                "paid_useful_work_status_base": puw_latest.get("puw_status", "INSUFFICIENT_DATA"),
                "paid_useful_work_completeness_base": puw_latest.get("puw_data_completeness", np.nan),
                "paid_useful_work_confidence_base": puw_latest.get("puw_confidence", np.nan),
                "ai_economic_coverage_base": cov_latest.get("ai_economic_coverage", np.nan),
                "ai_coverage_change_yoy_base": cov_latest.get("ai_coverage_change_yoy", np.nan),
                "ai_coverage_basis_base": cov_latest.get("ai_coverage_basis", "unavailable"),
                "catchup_status_base": cov_latest.get(
                    "catchup_status", "INSUFFICIENT_AI_DISCLOSURE"
                ),
                "ai_input_confidence_base": cov_latest.get("ai_input_confidence", np.nan),
                "cohort_irr_base": irr_latest.get("cohort_irr", np.nan),
                "cohort_irr_status_base": irr_latest.get(
                    "irr_status", "INSUFFICIENT_AI_DISCLOSURE"
                ),
                "manual_metrics_available": ",".join(manual_metrics),
                "manual_metric_count": len(manual_metrics),
            }
        )
    return pd.DataFrame(rows)


def flatten_recent_filings(submissions: Mapping[str, Any]) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, Mapping) or not recent:
        return pd.DataFrame()
    lengths = [len(v) for v in recent.values() if isinstance(v, list)]
    if not lengths:
        return pd.DataFrame()
    n = min(lengths)
    rows = []
    for i in range(n):
        rows.append({key: values[i] for key, values in recent.items() if isinstance(values, list) and len(values) > i})
    return pd.DataFrame(rows)


def sentence_snippets(text: str, phrases: Sequence[str]) -> list[tuple[str, str]]:
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    clean = html.unescape(soup.get_text(" "))
    clean = re.sub(r"\s+", " ", clean)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sentence in sentences:
        stripped = sentence.strip()
        if not 40 <= len(stripped) <= 1500:
            continue
        lower = stripped.lower()
        for phrase in phrases:
            if phrase.lower() in lower:
                key = re.sub(r"\W+", "", stripped.lower())[:250]
                if key not in seen:
                    seen.add(key)
                    results.append((phrase, stripped))
                break
    return results


def collect_filing_snippets(
    client: SECClient,
    universe: Sequence[Mapping[str, Any]],
    keyword_groups: Mapping[str, Sequence[str]],
    filings_per_company: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for company in universe:
        ticker = str(company["ticker"]).upper()
        cik = company["cik"]
        try:
            submissions = client.submissions(cik)
            filings = flatten_recent_filings(submissions)
            if filings.empty:
                continue
            filings = filings[filings["form"].isin(["10-Q", "10-K"])]
            filings = filings.sort_values("filingDate", ascending=False).head(filings_per_company)
            for filing in filings.to_dict("records"):
                primary_document = filing.get("primaryDocument")
                accession = filing.get("accessionNumber")
                if not primary_document or not accession:
                    continue
                text, source_url = client.filing_html(cik, accession, primary_document)
                for group_name, phrases in keyword_groups.items():
                    for matched_phrase, snippet in sentence_snippets(text, list(phrases)):
                        rows.append(
                            {
                                "ticker": ticker,
                                "company": company.get("name", ""),
                                "layer": company.get("layer", ""),
                                "filing_date": filing.get("filingDate"),
                                "report_date": filing.get("reportDate"),
                                "form": filing.get("form"),
                                "accession_number": accession,
                                "keyword_group": group_name,
                                "matched_phrase": matched_phrase,
                                "snippet": snippet,
                                "source_url": source_url,
                            }
                        )
        except Exception as exc:
            rows.append(
                {
                    "ticker": ticker,
                    "company": company.get("name", ""),
                    "layer": company.get("layer", ""),
                    "filing_date": pd.NaT,
                    "report_date": pd.NaT,
                    "form": "ERROR",
                    "accession_number": "",
                    "keyword_group": "pipeline_error",
                    "matched_phrase": "",
                    "snippet": str(exc),
                    "source_url": "",
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    paths: ProjectPaths,
    sec_quarters: pd.DataFrame,
    manual_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    puw_scores: pd.DataFrame,
    cohort_irr: pd.DataFrame,
    ecosystem_summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    snippets: pd.DataFrame,
) -> None:
    paths.output.mkdir(parents=True, exist_ok=True)
    tables = {
        "sec_quarterly_financials": sec_quarters,
        "manual_ai_metric_summary": manual_summary,
        "ai_economic_coverage": coverage,
        "paid_useful_work_scores": puw_scores,
        "capex_cohort_irr": cohort_irr,
        "ecosystem_summary": ecosystem_summary,
        "public_investor_scorecard": scorecard,
        "filing_evidence_snippets": snippets,
    }
    for name, frame in tables.items():
        frame.to_csv(paths.output / f"{name}.csv", index=False)

    metadata = {
        "generated_at_utc": utc_now_iso(),
        "app_version": APP_VERSION,
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
    }
    (paths.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if duckdb is not None:
        database_path = paths.output / "ai_public_investor_monitor.duckdb"
        con = duckdb.connect(str(database_path))
        try:
            for name, frame in tables.items():
                if frame.shape[1] == 0:
                    con.execute(
                        f'CREATE OR REPLACE TABLE "{name}" AS '
                        "SELECT CAST(NULL AS VARCHAR) AS _empty WHERE FALSE"
                    )
                    continue
                con.register("_frame", frame)
                con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _frame')
                con.unregister("_frame")
            metadata_frame = pd.DataFrame([metadata])
            con.register("_metadata", metadata_frame)
            con.execute("CREATE OR REPLACE TABLE run_metadata AS SELECT * FROM _metadata")
            con.unregister("_metadata")
        finally:
            con.close()


def load_cached_sec_outputs(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.output / "sec_quarterly_financials.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df


def download_sec_financials(
    client: SECClient,
    universe: Sequence[Mapping[str, Any]],
    history_years: int,
) -> pd.DataFrame:
    parser = CompanyFactsParser(history_years=history_years)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for company in universe:
        ticker = str(company["ticker"]).upper()
        try:
            facts = client.company_facts(company["cik"])
            parsed = parser.parse_company(
                facts,
                ticker=ticker,
                configured_name=str(company.get("name", ticker)),
                layer=str(company.get("layer", "")),
            )
            if not parsed.empty:
                frames.append(parsed)
            else:
                errors.append(f"{ticker}: no standardized facts found")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
    if errors:
        print("SEC warnings:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_pipeline(
    config_path: Path,
    user_agent: str | None,
    refresh_sec: bool,
    with_snippets: bool,
) -> dict[str, pd.DataFrame]:
    config = load_yaml(config_path)
    paths = ProjectPaths.from_config(config_path, config)
    paths.ensure()
    create_templates(paths)
    universe = config.get("universe", [])
    if not isinstance(universe, list) or not universe:
        raise ValueError("config.yaml must contain a non-empty universe list")

    manual = load_manual_metrics(paths.data / "manual_ai_metrics.csv")
    assumptions = load_assumptions(paths.data / "cohort_assumptions.csv")
    sec_quarters = pd.DataFrame()
    snippets = pd.DataFrame()

    need_client = refresh_sec or with_snippets
    client: SECClient | None = None
    if need_client:
        sec_cfg = config.get("sec", {})
        resolved_user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT")
            or str(sec_cfg.get("user_agent", ""))
        )
        client = SECClient(
            user_agent=resolved_user_agent,
            cache_dir=paths.cache,
            max_requests_per_second=float(sec_cfg.get("max_requests_per_second", 4.0)),
            cache_hours=float(sec_cfg.get("cache_hours", 12.0)),
            timeout_seconds=int(sec_cfg.get("timeout_seconds", 45)),
            retries=int(sec_cfg.get("retries", 4)),
        )

    if refresh_sec:
        assert client is not None
        sec_quarters = download_sec_financials(
            client,
            universe,
            history_years=int(config.get("analysis", {}).get("history_years", 7)),
        )
    else:
        sec_quarters = load_cached_sec_outputs(paths)
        if sec_quarters.empty:
            raise FileNotFoundError(
                "No cached SEC output. Run with --refresh-sec and a valid SEC User-Agent first."
            )

    if with_snippets:
        assert client is not None
        snippets = collect_filing_snippets(
            client,
            universe,
            config.get("keyword_groups", {}),
            filings_per_company=int(
                config.get("analysis", {}).get("snippet_filings_per_company", 2)
            ),
        )
    else:
        old_snippets = paths.output / "filing_evidence_snippets.csv"
        if old_snippets.exists():
            try:
                snippets = pd.read_csv(old_snippets)
            except pd.errors.EmptyDataError:
                snippets = pd.DataFrame()

    entity_lookup = entity_lookup_from_config(config)
    coverage = build_ai_coverage(sec_quarters, manual, assumptions, config)
    puw_scores = build_paid_useful_work_scores(manual, entity_lookup)
    cohort_irr = build_cohort_irr_table(coverage, assumptions, config)
    ecosystem_summary = build_ecosystem_summary(
        sec_quarters, coverage, puw_scores, config
    )
    manual_summary = latest_manual_summary(manual)
    scorecard = build_public_scorecard(
        sec_quarters, coverage, cohort_irr, manual, puw_scores, entity_lookup
    )
    save_outputs(
        paths,
        sec_quarters,
        manual_summary,
        coverage,
        puw_scores,
        cohort_irr,
        ecosystem_summary,
        scorecard,
        snippets,
    )
    return {
        "sec_quarters": sec_quarters,
        "manual_summary": manual_summary,
        "coverage": coverage,
        "puw_scores": puw_scores,
        "cohort_irr": cohort_irr,
        "ecosystem_summary": ecosystem_summary,
        "scorecard": scorecard,
        "snippets": snippets,
    }


def print_summary(outputs: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    scorecard = outputs.get("scorecard", pd.DataFrame())
    print(f"\nAI public-investor monitor completed. Outputs: {output_dir}")
    if scorecard.empty:
        print("No scorecard rows were produced.")
        return
    display_cols = [
        "ticker",
        "revenue_growth_yoy",
        "capex_growth_yoy",
        "downstream_metric_yoy_growth",
        "paid_useful_work_score_base",
        "ai_economic_coverage_base",
        "catchup_status_base",
        "cohort_irr_base",
        "cohort_irr_status_base",
    ]
    display_cols = [c for c in display_cols if c in scorecard.columns]
    print(scorecard[display_cols].to_string(index=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track whether public AI monetization is catching upstream capex."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init", "run", "build"],
        default="run",
        help="init creates input templates; run refreshes SEC data; build uses cached SEC data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help='SEC User-Agent, for example: "Your Name your@email.com"',
    )
    parser.add_argument(
        "--with-snippets",
        action="store_true",
        help="Download recent 10-K/10-Q filings and collect keyword evidence snippets.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    paths = ProjectPaths.from_config(args.config, config)
    paths.ensure()
    create_templates(paths)
    if args.command == "init":
        print(f"Templates created in {paths.data}")
        return 0
    try:
        outputs = build_pipeline(
            args.config,
            user_agent=args.user_agent,
            refresh_sec=args.command == "run",
            with_snippets=args.with_snippets,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print_summary(outputs, paths.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
