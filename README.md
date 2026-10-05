# Raw Quote Engine

Turn raw dealer quotes and transaction-level training data into a reproducible research report. The engine runs a fixed sequence: population diagnosis → exact-event history → causal quote features → quality and case diagnostics → chronological LightGBM experiments and model comparisons.

It is adapted from a BondCliQ research project, but requires no BondCliQ files, existing caches, saved models, or old notebook outputs. Quotes can be **wide bid/ask** or **long side/value**, in **spread or price units**. The supplied transaction bonds define the universe. The input period is not hardcoded.

## Start here

```bash
python -m pip install -e ".[notebook]"
python -m raw_quote_engine.cli demo --output runs/synthetic_demo
```

Open `runs/synthetic_demo/report.html`. This smoke run is explicitly synthetic and demonstrates the complete workflow; its gains are not evidence about any real quote feed.

For real data, copy [examples/config.json](examples/config.json), edit the file paths and column names, then run:

```bash
python -m raw_quote_engine.cli run --config examples/config.json
```

The default produces a **validation report** and reserves test. After reviewing the fixed candidate choice:

```bash
python -m raw_quote_engine.cli final-test examples/runs/my_research
```

Alternatively, add `--final-test` to the initial run command to execute that same sequence automatically: freeze the validation winner first, then evaluate only it and Base on test.

## DataFrame interface

```python
# CONFIGURATION LOGIC: Map the actual source columns; naive timestamps use the declared local timezone.
from raw_quote_engine import TransactionColumns, QuoteColumns, PipelineConfig, run_research

config = PipelineConfig(
    transactions=TransactionColumns(
        bond="CUSIP", time="prediction_time", target="BM_SPREAD",
        quantity="QUANTITY", issuer="ISSUER", sector="SECTOR",
        maturity_years="YRS_TO_MATURITY", anchor="PREV_BM_SPREAD", cpp="CPP_SPREAD",
    ),
    quotes=QuoteColumns(
        bond="cusip", known_time="known_time", dealer="dealer_id",
        bid="bid", ask="ask", quantity="size",
    ),
    value_kind="spread", unit="bps", timezone="America/New_York",
)
# ORCHESTRATION LOGIC: All five stages use the supplied data and save their actual calculated results.
run = run_research(data_ig, raw_quotes, base_features, lgbm_params, config, output="runs/my_research")
# UI LOGIC: Compare saved models, choose slices and export figures without another fit.
run.review()
```

The column names above are examples, not required source names. Omit an optional mapping when that metadata is unavailable. Bond ID, prediction/known time, quote dealer, quote values, target, and at least one base feature are essential. **Price and spread are not interchangeable**: supply compatible numerical units or perform a justified conversion upstream.

## What you receive

- A self-contained English report with calculated tables and complete PNG figures.
- Full-precision CSV diagnostics and a feature dictionary.
- Causal per-transaction features and saved prediction Parquet files.
- Four fixed experiments: Base, Quote, Quote+Path, Quote+CrossBond. The latter adds matched-dealer and other-bond issuer changes; it is not a fitted latent-factor model.
- MAE, RMSE, P95, bias, paired win rate, coverage, date stability, default one-/two-column slices, and arbitrary model-pair comparisons.
- Priority slices for large trades, maturity, historical activity and anchor-versus-CPP discrepancies when the required metadata exists.
- Native LightGBM models, training category schemas, frozen selection, input fingerprints, package versions and resumable local stage caches.

State calculations retain zero/negative spreads, unknown sizes, multi-price events and crossings. They never silently drop transactions without quote coverage. Raw candidate clipping/downweighting is a diagnosed feature alternative, not a destructive cleaning operation.

Read [the usage guide](docs/USAGE.md), [the data contract](docs/DATA_CONTRACT.md), and [the BondCliQ lessons](docs/BONDCLIQ_LESSONS.md). Open [raw_quote_research.ipynb](raw_quote_research.ipynb) for a notebook entry point. No QR transfer or camera workflow is needed.
