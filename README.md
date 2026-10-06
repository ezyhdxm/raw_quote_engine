# Raw Quote Engine

Compare raw dealer **spread quotes** with a user-supplied transaction baseline. The engine runs population diagnosis, exact-event history, causal quote features, quality/case diagnostics, and chronological LightGBM comparisons. All report tables and figures are calculated from the current inputs.

It accepts prepared transaction and quote DataFrames, an ordered baseline feature list, an explicit categorical-feature list and LightGBM parameters. Source mappings and research settings are editable in a notebook form. Quotes may be wide bid/ask or long side/value. Price input is rejected; any price-to-spread conversion belongs upstream.

## Start in a notebook

```bash
python -m pip install -e ".[notebook]"
python -m jupyter lab raw_quote_research.ipynb
```

[raw_quote_research.ipynb](raw_quote_research.ipynb) starts with clearly labelled synthetic inputs. Replace them with your prepared DataFrames, inspect the form, click **Validate inputs**, then **Run validation**. Running all cells alone does not train or open final test.

The notebook forms use responsive sections, expandable advanced fields and badges that distinguish pending edits from saved results. Comparisons require **Apply**; exports use the last successful applied settings. HTML reviews share the same visual style, with section navigation, sticky table headers and scrollable full-resolution figures.

For the existing BondCliQ files, use [bondcliq_research.ipynb](bondcliq_research.ipynb) and [its guide](docs/BONDCLIQ_EXAMPLE.md). Its source preparation and configuration are visible in notebook cells. The baseline now uses separate counterparty and side categories, giving 15 inputs; it does not replay historical BASE14 scores. There is no dataset-specific engine preset.

## DataFrame interface

```python
# SETUP LOGIC: Import the generic form; inputs and upstream feature definitions remain user-owned.
from raw_quote_engine import research_form

# UI LOGIC: Supply the five inputs, map source columns in the form, then validate and start explicitly.
controller = research_form(
    transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
    output='runs/my_research',
)
```

An optional `config=PipelineConfig(...)` and `training=TrainingConfig(...)` prefill the form. The engine does not infer sector mappings, synthesize a proxy, engineer the supplied baseline, repair source multipliers or guess benchmark conventions. After validation completes, `controller.run` is the result; `controller.run.review()` opens comparisons of saved predictions without refitting.

For scripts, use the same explicit inputs and mappings:

```python
# SETUP LOGIC: Use the programmatic pipeline only when starting a run is intended.
from raw_quote_engine import run_research
# ORCHESTRATION LOGIC: This call starts Steps 1–5 through validation and saves their computed evidence.
run = run_research(
    transactions_df, quotes_df, BASE_FEATURES, LGB_PARAMS, config,
    base_cat_features=BASE_CAT_FEATURES, training=training, output='runs/my_research',
)
```

A CLI configuration is shown in [examples/config.json](examples/config.json). Its paths and columns are illustrative. For a synthetic smoke report: `python -m raw_quote_engine.cli demo --output runs/synthetic_demo`.

## Optional walk-forward validation

The form defaults to **Single split**. Select **Walk forward**, configure the minimum training history, validation block, embargo and optional rolling cap, then click **Validate inputs** to review exact fold dates, row counts and the fit budget. The final Test reservation still comes from the outer split; it never enters a CV fold.

Each fold fits fresh models and learns categories only from its training rows. An optional label-known-time column purges training labels unavailable at each fold origin and masks OOF outcomes still unavailable at the final selection origin. Pooled out-of-fold predictions select the candidate; Base and that candidate are then refitted on development history (using the rolling cap when configured). The budget is `4 × folds + 2` fits, while quote events and causal features are built once. Reports include per-fold metrics and stability; confidence intervals and significance are not inferred automatically. See [the walk-forward guide](docs/USAGE.md#optional-walk-forward-validation).

## What the run produces

- An English report, complete PNG figures, exact CSV diagnostics and a feature dictionary.
- Causal per-transaction features, saved predictions and native LightGBM models.
- Four fixed families: Base, Quote, Quote+Path and Quote+CrossBond. The last needs an issuer mapping for other-bond features; otherwise it adds same-bond movement only. No fitted latent-factor model is implied.
- Common-record MAE, RMSE, P95, bias, win rate, coverage and descriptive date stability.
- Any pair of saved models, arbitrary metadata slices, two-column heatmaps and custom exports.
- Optional quantity, maturity, history and prior-anchor/proxy slices when their input contracts are supplied.
- Frozen validation selection, input fingerprints, category schemas and resumable stage caches.

No-quote transactions, zero/negative spreads, unknown sizes, simultaneous candidates and crossings remain visible. Candidate clipping and dealer downweighting are diagnostics, not destructive source cleaning. Final test requires an explicit action after validation. Single-split runs reuse the Train fits; optional walk-forward runs refit only Base and the selected candidate on development history before final test.

Read [USAGE.md](docs/USAGE.md), [DATA_CONTRACT.md](docs/DATA_CONTRACT.md) and [BONDCLIQ_LESSONS.md](docs/BONDCLIQ_LESSONS.md). Source-specific preprocessing stays in the user's notebook; data, caches, models and generated reports stay local.
