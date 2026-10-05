# BondCliQ + prepared data_ig example

Open [bondcliq_research.ipynb](../bondcliq_research.ipynb). All BondCliQ-specific preparation, column choices, feature lists and estimator parameters are visible in this notebook and its [paired source](../examples/bondcliq_research.py). The engine has no dataset preset or implicit BondCliQ preparation.

## Run it

Download and extract the repository ZIP, then install from its directory:

```bash
python -m pip install -e ".[notebook]"
python -m jupyter lab bondcliq_research.ipynb
```

Git is not required. Use the same Python environment for the installation and notebook kernel. Edit `DATA_IG_FILE` and `QUOTES_FILE`; their defaults are:

```text
data/pipeline/data_ig.parquet
data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet
```

Run the notebook cells. They read and prepare the inputs, then open an editable form. Inspect the source mappings, baseline lists, parameter dictionary, date policy and output folder. Click **Validate inputs**, resolve any errors, then click **Run validation**. The form starts Steps 1–5 and exposes its completed `ResearchRun` as `controller.run`. Running all cells alone never starts training. After completion, run the review cell to open comparisons of saved predictions.

Use a fresh output folder for changed inputs or settings. Repeating an unchanged run reuses completed stages. The generated `report.html` contains this run's calculated tables and complete PNGs. No old predictions, hand-entered losses or research caches are required.

The last cell defaults to `RUN_FINAL_TEST = False`. Enable it only after accepting the validation choice, then run that cell. Final test evaluates the original Train-fitted Base and frozen candidate; it does not reproduce the old notebook's pre-test refit.

## What the user prepares

`data_ig` is the existing engineered transaction table, not a bare TRACE file. The user must supply production baseline features, target, anchor, actual spread, corrected identifiers, time, units and any relevant benchmark adjustments. The engine does not implement the production loader, build a CPP proxy, retrieve sector files, repair TRACE multipliers or infer trade direction.

Required transaction fields include the baseline list below and `CUSIP`, `ISSUER`, `EFFECTIVE_DATETIME_TS`, `D_BM_SPREAD` and `BM_SPREAD`. The raw quote file must contain `cusip`, `firm`, `side`, `spread`, `quantity` and `quote_timestamp_UTC`. For another schema, use the generic [form notebook](../raw_quote_research.ipynb).

Optional `SECTOR_MAP_FILE` is handled entirely in the example. It must contain `CUSIP` and `SECTOR`; explicit nonblank labels override matching transaction records. Conflicting map values reject, unmatched records retain their original sector, and no additional bond enters the universe. Set it to `None` to use existing metadata unchanged.

## Baseline and parameters

This example intentionally uses **15 baseline features**, replacing the old joint `CONTRA_PARTY_SIDE` field with separate `CONTRA_PARTY_TYPE` and `SIDE` inputs. `PREV_TRADE_TYPE` is retained. Consequently, this is a revised baseline, not a replay of historical BASE14 scores.

```text
D_CPP_BM_SPREAD
COUPON
D_CDX_TRADE
MEAN_ISSUER_SPREAD_DEV
NUM_OF_ISSUER_TRADES_SINCE_PREV
PREV_BM_SPREAD
PREV_QUANTITY
PREV_TRADE_TYPE
QUANTITY
CONTRA_PARTY_TYPE
SIDE
YRS_TO_MATURITY
BM_YIELD_STD
D_SHORT_TO_BM_SPREAD
PREV_BM_SPREAD_STD_GROUP_BY_TYPE
```

`BASE_CAT_FEATURES = ['PREV_TRADE_TYPE', 'CONTRA_PARTY_TYPE', 'SIDE']` is explicit. All other baseline inputs are numeric. The engine learns categorical vocabularies from Train only and treats unseen evaluation categories as missing. The example does not uppercase, combine or reinterpret the source category labels.

The LightGBM parameter dictionary remains the original one:

```python
# CONFIGURATION LOGIC: User-supplied estimator parameters are passed without engine defaults.
LGB_PARAMS = dict(
    objective='mae', boosting_type='dart', n_estimators=400, learning_rate=.2,
    num_leaves=127, max_bin=511, max_depth=-1, min_child_samples=20,
    min_split_gain=0., subsample=1., subsample_freq=0, colsample_bytree=1.,
    reg_alpha=0., reg_lambda=0., n_jobs=8, verbosity=-1, random_state=2026,
)
```

`category_order='appearance'` preserves training occurrence order; `apply_model_defaults=False` leaves unspecified parameters to LightGBM. Library versions and thread behavior can still affect exact fitted results.

## Target, anchor and optional rollover adjustment

The supplied target is unchanged. When `D_BM_YIELD_OFFSET` exists, the example maps it as `rollover_adjustment` and expects the upstream convention:

```text
target = actual spread - anchor + adjustment
predicted spread = predicted target + anchor - adjustment
```

For example, target prediction `0.03`, anchor `1.00` and adjustment `0.02` reconstruct spread `1.01`. Against actual `1.02`, the error is `-1 bp`. Scoring always uses explicit `BM_SPREAD`; it does not replace observed truth with target plus anchor. Without a mapped adjustment, reconstruction uses zero adjustment. Verify that the supplied delta uses this sign convention; do not map an unrelated benchmark field.

This scalar correction aligns the transaction's prediction baseline. It does **not** repair benchmark changes across earlier quote timestamps, reconstruct a historical Treasury curve or establish that two vendor spread definitions are economically identical. Those conventions require upstream preparation.

The optional proxy mapping uses `MID_SPREAD_CPP` only when present. There is no computed proxy, bid/ask fallback or rowwise substitution. The diagnostic compares the proxy with the adjusted anchor, `PREV_BM_SPREAD - adjustment`. Missing proxy values remain unsupported. A mapped missing/nonfinite adjustment rejects the run rather than silently substituting zero. Pretrade discrepancy is a diagnostic signal, not proof of an erroneous previous trade.

## Time, universe and scale

| Setting | Explicit example value |
|---|---|
| Universe | All CUSIPs in the original supplied `data_ig`, before model-label filtering |
| Intended history | Supply the same three-month snapshot yourself; no date range is fetched or assumed |
| Quote known time | `quote_timestamp_UTC`; naive values mean UTC |
| Transaction time and state date | `EFFECTIVE_DATETIME_TS`; naive values mean America/New_York |
| Quote value | Spread only, raw bps multiplied by 0.01 into percentage points |
| Transaction values | Target, actual, anchor, optional proxy/adjustment remain percentage points |
| Reported error | Spread difference multiplied by 100, in bps |
| Freshness / lookback / synchronization | 30 / 30 / 1 minutes; exact known-time quotes allowed |
| Quantity scaling | No added multiplier on either source |
| Eligibility | Finite target, anchor and truth; valid CUSIP/time; no quote-coverage exclusion |
| Experiment end | End of final local quote date within the full supplied universe |
| Evaluation dates | Final 5 observed eligible dates Test; prior 5 Validation; 2 Train embargo dates; no extra Test embargo |
| Minimum history | At least 10 earlier training dates |

The notebook explicitly records `excluded_rows` during preparation and preserves `quote_universe` separately. A bond with an invalid model target can still belong to the full quote population. Dates count observed transaction days, not calendar-day subtraction. Unknown sizes, zero/negative spreads, simultaneous candidates and crossing are retained for diagnosis.

If `TRADE_COUNTS_PREV_MONTH` exists, the example explicitly maps that field for priority history slices. Otherwise the generic engine can use the supplied transaction history, with coverage limitations disclosed; a trailing 30-day count is not the same as the prior-calendar-month count. Quote-size units remain a user responsibility and are not assumed executable or comparable to transaction par simply because values are numeric.

## Results to inspect

The fixed families are Base, Quote, Quote+Path and Quote+CrossBond. They differ from the old Quote levels / Reliability / Age decay chain. Review >=1MM trades, poor historical liquidity and prior-anchor/proxy discrepancies above 5 and 10 bps; separate short maturity and missing metadata. Any improvement is measured against the newly trained BASE15 on common records. A 5% practical threshold is not a significance test.

Every table and chart comes from the supplied run. Synthetic verification checks software behavior and input contracts; it says nothing about real-data predictive gain. See [USAGE.md](USAGE.md) for custom slices, any pair of saved models, output artifacts and resuming completed work.
