# BondCliQ + data_ig: a ready-to-run research example

Use [bondcliq_research.ipynb](../bondcliq_research.ipynb) to run Steps 1–5 with the BondCliQ source mapping, the ordered 14 baseline features, and the LightGBM parameters from the existing Step 5 notebook. Change only the two input paths to start. This is a fresh engine experiment with the same baseline and prediction task; its quote candidate families and final-test fitting policy differ from the historical notebook, as described below.

## 1. Install and open the notebook

Download the repository ZIP and extract it, or update your existing checkout. From a terminal in the repository directory, run:

```bash
python -m pip install -e ".[notebook]"
python -m jupyter lab bondcliq_research.ipynb
```

Use the same Python environment for installation and the notebook kernel. On Windows, the commands work in a terminal with Python available; Git is not required. The default files are:

```text
data/pipeline/data_ig.parquet
data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet
```

These files are not included in the repository. You can place them there or change `DATA_IG_FILE` and `QUOTES_FILE` to absolute paths, such as `Path(r"D:\research\data_ig.parquet")`. Keep the existing data files unchanged.

Run the notebook cells in order. The main cell performs normalization, Steps 1–4 diagnostics/features, four validation fits, selection and report export. It reports stage progress and reuses completed caches. Open `runs/bondcliq/report.html` for the generated English report; the notebook review panel reuses saved predictions.

The final cell has `RUN_FINAL_TEST = False`. Change it to `True` and run that cell only after accepting the frozen validation choice. It calls `run.finalize_test()` and updates the report with final-test evidence. Nothing in the initial run evaluates test losses.

## 2. Equivalent Python call

```python
# SETUP LOGIC: Import the dataset-specific adapter over the shared raw-quote pipeline.
from raw_quote_engine.presets.bondcliq import run_bondcliq

# ORCHESTRATION LOGIC: Read the supplied files, run all stages, and save computed validation evidence.
run = run_bondcliq(
    "data/pipeline/data_ig.parquet",
    "data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet",
    output="runs/bondcliq",
)
# UI LOGIC: Inspect saved paired predictions without fitting another model.
panel = run.review(stage="Validation")
```

The first two arguments can instead be pandas DataFrames, so `run_bondcliq(data_ig, bcq_df)` works with existing in-memory inputs. Optional `sector_map` accepts a file/DataFrame with `CUSIP` and `SECTOR`; explicit mapped values override the existing sector for matching CUSIPs, while unmatched rows keep their source metadata. It does not add bonds to the modeling population. Optional `cache_dir` places reusable stage caches in a chosen local directory; `progress` accepts a `(stage, done, total, detail)` callback.

## 3. What must already be in data_ig

`data_ig` means the existing engineered transaction table, not a bare TRACE print file. It must already contain the baseline fields below, plus `CUSIP`, `ISSUER`, `EFFECTIVE_DATETIME_TS`, `BM_SPREAD`, `D_BM_SPREAD`, `CONTRA_PARTY_TYPE` and `SIDE`. `CONTRA_PARTY_SIDE` is created by the adapter. `SECTOR` and existing CPP/liquidity metadata enable their corresponding diagnoses.

The raw BondCliQ file must contain `cusip`, `firm`, `side`, `spread`, `quantity` and `quote_timestamp_UTC`. This preset uses the existing long side/value format. Other schemas, including wide bid/ask, should use the generic column-mapping interface in [USAGE.md](USAGE.md).

The adapter does not reproduce the production pipeline that creates rolling issuer features, yield/benchmark adjustments or historical trade fields. If starting with unengineered prints, first run the established production loader that creates `data_ig`, then use this notebook. No original research caches, saved models or previously rendered results are needed after that.

The fixed ordered `BASE_FEATURES` list is:

```text
 1. D_CPP_BM_SPREAD
 2. COUPON
 3. D_CDX_TRADE
 4. MEAN_ISSUER_SPREAD_DEV
 5. NUM_OF_ISSUER_TRADES_SINCE_PREV
 6. PREV_BM_SPREAD
 7. PREV_QUANTITY
 8. PREV_TRADE_TYPE
 9. QUANTITY
10. CONTRA_PARTY_SIDE
11. YRS_TO_MATURITY
12. BM_YIELD_STD
13. D_SHORT_TO_BM_SPREAD
14. PREV_BM_SPREAD_STD_GROUP_BY_TYPE
```

`PREV_TRADE_TYPE` and `CONTRA_PARTY_SIDE` are categorical; other baseline columns are coerced to numeric. The current trade category is the unambiguous JSON pair of the supplied counterparty and side: `C` / `B` becomes `'["C","B"]'`, while `D` / missing becomes `'["D",null]'`. The adapter does not infer dealer/customer direction, use the unreliable current `TRADE_TYPE`, or rewrite the meaning of `PREV_TRADE_TYPE`.

## 4. Exact training parameters and task

The preset exposes `BASE_FEATURES`, `LGB_PARAMS` and `prepare_bondcliq_inputs` for inspection. The notebook imports them rather than maintaining a second executable copy of these settings.

```python
# CONFIGURATION LOGIC: These are the existing Step 5 LightGBM parameters, not a tuned replacement.
LGB_PARAMS = dict(
    objective="mae", boosting_type="dart", n_estimators=400, learning_rate=0.2,
    num_leaves=127, max_bin=511, max_depth=-1, min_child_samples=20,
    min_split_gain=0.0, subsample=1.0, subsample_freq=0,
    colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
    n_jobs=8, verbosity=-1, random_state=2026,
)
```

Every model learns `D_BM_SPREAD`. Its predicted level is `predicted_delta + PREV_BM_SPREAD`, and the scoring truth is the supplied **`BM_SPREAD`**. Do not replace scoring truth with `D_BM_SPREAD + PREV_BM_SPREAD`: the historical delta may include a roll adjustment. Both level values are in percentage points; multiplying their difference by 100 gives basis points. For example, predicted delta `0.02`, anchor `0.60` and actual `0.63` give predicted level `0.62` and error `-1 bp`.

The preset passes the displayed LightGBM parameters without adding the generic engine's deterministic training overrides. Library version, thread behavior and revised engine features can still affect fitted results; the manifest records the effective configuration.

## 5. Universe, time, units and dates

| Setting | BondCliQ preset |
|---|---|
| Quote universe | Only CUSIPs appearing in the original supplied `data_ig`; no full-feed universe expansion |
| Research history | Supply the same three-month `data_ig` table used previously; the adapter does not invent or fetch missing history |
| Quote known time | `quote_timestamp_UTC`, interpreted as UTC and converted to `America/New_York` |
| Transaction time | `EFFECTIVE_DATETIME_TS`; naive timestamps mean New York local time |
| Event timing | Exact known time; quotes known at the transaction time are allowed |
| State | Same local date; freshness 30 minutes; bid/ask synchronization 1 minute |
| Quote spreads | Convert raw bps to percentage points with factor `0.01`; zero and negative spreads remain |
| Quantity | No new multiplier on either source; missing/zero/negative quote sizes remain distinct |
| Experiment end | Final valid quote day in the original transaction CUSIP universe |
| Date allocation | Last 5 observed eligible trade dates = Test; preceding 5 = Validation; preceding 2 = training embargo; at least 10 earlier training dates |
| Rows without quotes | Retained in the eligible modeling population |

The date allocation uses observed New York trading dates, not calendar-day subtraction. It intentionally does not use the generic engine's default fractional split or insert another embargo between Validation and Test. The preset audits invalid target/key exclusions and stops when insufficient dates remain. If you supply a different period, these counts stay fixed, so a very short input may fail the ten-training-date requirement.

Quote quantity units are not assumed to equal transaction par units merely because both are numeric. Preserving the original quote sizes supports within-quote size diagnostics; it does not certify transaction-size comparability. Multi-price events, crossings and unknown size remain diagnostic information rather than automatic deletion rules. Existing TRACE and multiplier handling in `data_ig` is preserved.

For the prior-anchor audit, the preset uses the available CPP representation in this order: `MID_SPREAD_CPP + D_BM_YIELD_OFFSET`; otherwise the mean of `D_CPP_BM_SPREAD_BID` and `D_CPP_BM_SPREAD_ASK` plus `PREV_BM_SPREAD`; otherwise `D_CPP_BM_SPREAD + PREV_BM_SPREAD` as the documented legacy proxy. It records which source was used. The proxy is not presented as a newly observed quote.

`CPP_AUDIT_GAP_BPS` retains the historical discrepancy arithmetic directly. In the delta fallback, an exact `0.05` or `0.10` becomes 5 or 10 bps without adding then subtracting the anchor; it does not enter the corresponding strict `>5` or `>10` cohort through cancellation rounding. These audit fields are not added to BASE14.

## 6. What to review, and what differs from the old experiment

Start with current trades of at least 1MM, especially those with prior-anchor/CPP gaps above 5 bps or 10 bps, and weak historical liquidity. The liquidity priority uses `TRADE_COUNTS_PREV_MONTH` when supplied; absent history is disclosed rather than inferred from future trades. Review maturity below one year separately from the priority population. The report also includes coverage, sector/issuer/dealer diagnostics, representative cases, daily stability and slice support. A 5% loss reduction is a practical threshold; it is not a statistical-significance test.

The shared engine fits **Base, Quote, Quote+Path and Quote+CrossBond**. These differ from the old Step 5 chain **Base, Quote levels, Reliability and Age decay**. New quote feature definitions and diagnostics are those of the reusable engine. Final test reuses the train-only fitted Base and validation-selected candidate; it does not reproduce the historical notebook's pre-test refit. Accordingly, this example reproduces the original baseline columns, parameters, target/anchor/truth, source mappings and pilot split, **not bit-for-bit historical model gains**.

All result tables and figures are calculated from this run. The repository ships no hand-entered real-data losses or precomputed BondCliQ result images. Local verification on synthetic inputs tests plumbing and contracts, not prediction benefit on the real feed.

## 7. Resume and outputs

Repeating the same main call reuses completed work when inputs/settings match. Use a new `output` directory for a changed experiment; a shared `cache_dir` can reuse eligible stages. To review an existing run without rereading raw files:

```python
# FILE IO LOGIC: Restore saved features and predictions from your trusted local run.
from raw_quote_engine import load_run
run = load_run("runs/bondcliq")
# UI LOGIC: Existing predictions supply all model and slice comparisons.
panel = run.review(stage="Validation")
```

Before expensive stages begin, `runs/bondcliq_preflight.json` records source counts, exclusions, missing sector/history metadata, CPP source, date settings and exact baseline parameters. On completion the same audit is saved inside the run as `bondcliq_preflight.json` and in the manifest. It describes preparation, not prediction benefit.

`report.html`, `tables/`, the feature dictionary, `features.parquet`, `predictions_validation.parquet`, native model files and the frozen `selection.json` live in the run directory. After explicit finalization, `predictions_test.parquet` and the updated report contain Base and the selected candidate only. See [USAGE.md](USAGE.md) for arbitrary slices, two-column heatmaps and comparing any two saved models.
