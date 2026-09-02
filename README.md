# AI Public-Investor Monitor

A practical Python pipeline for testing whether **paid downstream AI economics are catching up with upstream AI capital spending**.

It combines:

- SEC Company Facts for company-wide financial statements
- manually entered AI-specific disclosures from earnings releases, presentations, interviews, and private-company announcements
- low/base/upside estimates with a confidence score
- a transparent Paid Useful Work proxy score
- an AI capital-stock model
- an economic-coverage test
- an optional capex-cohort IRR model
- a Streamlit dashboard and CSV/DuckDB outputs

The pipeline intentionally returns `INSUFFICIENT_AI_DISCLOSURE` when the public evidence does not support an AI-specific conclusion. It does not infer private-company gross margins from token usage alone.

## 1. Quick start on Windows

Open Command Prompt or the VS Code terminal in this folder:

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SEC_USER_AGENT=Your Name your.email@example.com
python ai_monitor.py run --user-agent "%SEC_USER_AGENT%"
streamlit run dashboard.py
```

To also search the latest 10-K and 10-Q filings for capex, financing, demand, and monetization language:

```bat
python ai_monitor.py run --user-agent "%SEC_USER_AGENT%" --with-snippets
```

After the first SEC download, rebuild the model without making network requests:

```bat
python ai_monitor.py build
```

Run the tests:

```bat
python -m unittest discover -s tests -v
```

## 2. What is automatic and what is manual

### Automatic from SEC Company Facts

For each public company in `config.yaml`, the pipeline attempts to retrieve:

- revenue
- gross profit
- operating income
- net income
- operating cash flow
- capital expenditures
- depreciation and amortization
- research and development expense
- share-based compensation
- accounts receivable
- net property, plant, and equipment
- cash and total assets

It converts cumulative year-to-date XBRL cash-flow facts into standalone quarters, then calculates TTM values, year-over-year growth, free-cash-flow proxy, capex intensity, receivable trends, and incremental gross profit per incremental capex.

### Manual AI-specific disclosures

Public filings usually do not identify the exact portion of capex, revenue, or contribution profit attributable to AI. Add those observations to:

```text
data/manual_ai_metrics.csv
```

Use actual dollars, not “dollars in millions.” For example, enter `14000000000` or `14e9` for $14 billion. Ratios can be entered as either `0.35` or `35`; the model interprets both as 35%.

## 3. Paid Useful Work proxy

Because a public investor cannot observe task-level usefulness across OpenAI, Anthropic, Palantir, or their customers, the pipeline builds a **0-100 proxy score** rather than pretending to know an exact number of economically useful tokens.

The score uses only the components you supply:

| Component | Weight | Typical public proxy |
|---|---:|---|
| Paid demand growth | 25% | AI revenue, ARR, RPO, paid customers, or paid-token growth |
| Production adoption | 20% | Production usage share |
| Task success | 15% | Acceptance, autonomous resolution, or successful completion rate |
| Persistence | 20% | Renewal rate or net revenue retention |
| Customer value growth | 10% | Customer ROI or value-created index |
| Contribution margin | 10% | AI contribution margin |

Missing components are not silently treated as zero. The code renormalizes available weights, reports data completeness, and refuses to issue a score unless at least 35% of the component weight is observable. Paid-token growth is capped as a weaker signal because more tokens do not necessarily mean more useful work.

The ecosystem summary keeps two sides separate:

- downstream Paid Useful Work score for model and application companies
- AI economic coverage for configured capital owners such as hyperscalers and cloud providers

It does not add revenue across chips, clouds, model providers, and applications, which would double count the same end-customer dollar.

## 4. Minimum inputs for AI economic coverage

For a public company, enter:

1. Either `ai_capex_share` or `ai_capex_dollars`
2. One of:
   - `ai_contribution_profit`
   - `ai_contribution_profit_run_rate`
   - `ai_revenue` plus `ai_contribution_margin`
   - `ai_revenue_run_rate` plus `ai_contribution_margin`
   - `ai_incremental_gp_attribution_share`

The model then estimates:

```text
AI economic coverage
= AI contribution profit
  / (economic depreciation + required return on AI capital + maintenance charge)
```

It uses TTM AI contribution profit when enough history exists. When only a current run-rate disclosure is available, it clearly labels the result `annualized_current_run_rate` rather than pretending that four quarters were reported.

Interpretation used by the code:

- `SUPPORTED`: coverage at least 1.25x and not materially deteriorating
- `WATCH`: coverage between roughly 0.8x and 1.25x, or mixed evidence
- `NOT_SUPPORTED`: coverage below 0.8x and not improving
- `INSUFFICIENT_AI_DISCLOSURE`: not enough evidence or confidence

These thresholds are screening rules, not accounting standards.

## 5. Manual metric file

Columns in `manual_ai_metrics.csv`:

| Column | Meaning |
|---|---|
| `as_of_date` | Period or date represented by the metric |
| `ticker` | Public ticker or private-company identifier such as `OPENAI` or `ANTHROPIC` |
| `company` | Company name |
| `metric` | Standardized metric name |
| `value` | Base estimate |
| `low` | Downside bound |
| `high` | Upside bound |
| `unit` | `USD`, `ratio`, `count`, or `index` |
| `period_type` | `quarter`, `annual`, `ttm`, `run_rate`, or `point_in_time` |
| `source_name` | Filing, earnings release, interview, presentation, or article title |
| `source_url` | Source link |
| `source_date` | When the source became public |
| `confidence` | Number from 0 to 1 |
| `reported_or_estimated` | `reported`, `derived`, or `modeled` |
| `notes` | Methodology and caveats |

Recognized metric names include:

| Metric | Use |
|---|---|
| `ai_revenue` | AI revenue for the stated period |
| `ai_revenue_run_rate` | Annualized AI revenue run rate |
| `ai_arr` | AI annual recurring revenue |
| `ai_rpo` | AI remaining performance obligations |
| `ai_contribution_profit` | AI contribution profit for the stated period |
| `ai_contribution_profit_run_rate` | Annualized AI contribution-profit run rate |
| `ai_contribution_margin` | AI revenue less directly attributable variable costs, divided by AI revenue |
| `ai_capex_share` | Estimated portion of reported company capex attributable to AI |
| `ai_capex_dollars` | Direct AI capex estimate |
| `ai_incremental_gp_attribution_share` | Estimated portion of incremental company gross profit caused by AI |
| `paid_useful_work_index` | Your normalized index of paid successful AI work |
| `paid_token_index` | Paid, cost-weighted token activity index |
| `production_share` | Production usage divided by total tracked usage |
| `renewal_rate` | Customer renewal rate |
| `net_revenue_retention` | Net revenue retention |
| `paid_enterprise_customers` | Paying enterprise customer count |
| `task_success_rate` | Accepted or successfully completed tasks divided by attempts |
| `customer_value_index` | Normalized customer ROI or value-created index |
| `gpu_utilization` | Utilization of deployed AI compute |
| `spot_gpu_price_index` | Rental-price index for comparable GPU capacity |
| `vendor_financing` | Vendor financing or similar support |
| `purchase_commitments` | Purchase and supply commitments |
| `inference_share` | Inference portion of AI compute or spend |

Example format only; replace the figures and sources with your own work:

```csv
as_of_date,ticker,company,metric,value,low,high,unit,period_type,source_name,source_url,source_date,confidence,reported_or_estimated,notes
2026-06-30,EXAMPLE,Example Co,ai_capex_share,0.60,0.45,0.75,ratio,point_in_time,Investor estimate,,2026-08-01,0.45,modeled,Range based on management capex commentary
2026-06-30,EXAMPLE,Example Co,ai_revenue_run_rate,1000000000,750000000,1300000000,USD,run_rate,Company disclosure,,2026-08-01,0.75,derived,Annualized from disclosed quarterly figure
2026-06-30,EXAMPLE,Example Co,ai_contribution_margin,0.35,0.20,0.50,ratio,point_in_time,Investor estimate,,2026-08-01,0.35,modeled,Revenue less direct inference and support costs
```

The downside scenario uses lower beneficial metrics and higher adverse metrics. For example, it uses the low AI margin but the high AI-capex estimate.

## 6. Cohort IRR assumptions

Optional overrides belong in:

```text
data/cohort_assumptions.csv
```

Use one row per ticker and scenario. A ticker of `DEFAULT` applies to companies without a specific override.

Key assumptions:

- `revenue_per_capital_at_full_utilization`
- `contribution_margin`
- year-one, year-two, and mature utilization
- useful economic life
- annual price change
- annual efficiency captured by the capital owner
- maintenance capex
- tax rate
- salvage value
- hurdle rate

When possible, the code estimates current revenue per gross AI capital from your AI-revenue input and its modeled AI capital stock. An explicit assumption overrides that estimate.

## 7. Main outputs

The pipeline writes the following files to `outputs/`:

| File | Contents |
|---|---|
| `sec_quarterly_financials.csv` | Standardized quarterly and TTM SEC financials |
| `manual_ai_metric_summary.csv` | Latest low/base/upside manual disclosures and growth |
| `ai_economic_coverage.csv` | AI capital stock, capital charge, contribution profit, and coverage by scenario |
| `paid_useful_work_scores.csv` | Paid demand, production, task success, persistence, customer value, margin, completeness, and composite score |
| `ecosystem_summary.csv` | Downstream Paid Useful Work versus capital-owner AI coverage |
| `capex_cohort_irr.csv` | New-capex cohort IRR sensitivities |
| `public_investor_scorecard.csv` | Latest company-level decision table |
| `filing_evidence_snippets.csv` | Keyword evidence from recent filings when enabled |
| `ai_public_investor_monitor.duckdb` | All output tables in one queryable database when DuckDB is installed |

Example DuckDB query:

```sql
SELECT
    ticker,
    revenue_growth_yoy,
    capex_growth_yoy,
    ai_economic_coverage_base,
    catchup_status_base,
    cohort_irr_base
FROM public_investor_scorecard
ORDER BY ai_economic_coverage_base DESC NULLS LAST;
```

## 8. How to read the result

The strongest evidence for a durable AI capex cycle is:

```text
paid downstream usage growing
+ production deployments and renewals improving
+ AI contribution margins holding up
+ AI economic coverage above 1.25x
+ new capex cohorts clearing the required return
+ receivables and vendor financing not deteriorating
```

Important counterexamples:

- Tokens can rise while useful work and customer ROI fall.
- AI applications can create substantial value while hardware demand slows because compute per task declines.
- Nvidia or another upstream supplier can grow while the capital owner earns a poor return.
- AI can be economically transformative while a public stock is still too expensive.

The pipeline answers an operating-capital question. Equity valuation still requires a separate expected-return model using the current share price.
