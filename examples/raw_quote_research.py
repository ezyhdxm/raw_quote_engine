# %% [markdown]
# # Raw spread quote research: DataFrames to report
#
# This notebook opens an editable setup form. Run the cells, inspect mappings and
# settings, click **Validate inputs**, then **Run validation**. Run All alone never trains.
# The default inputs are synthetic and labelled as such. Their gains demonstrate
# plumbing, not evidence about a real feed. Only spread-valued quotes are supported.

# %%
# SETUP LOGIC: Import the public form, mapping objects and local data reader.
from pathlib import Path
from html import escape
import pandas as pd
from IPython.display import HTML, display
from raw_quote_engine import PipelineConfig, TransactionColumns, QuoteColumns, TrainingConfig, research_form, Slice
from raw_quote_engine.demo import demo_inputs

# CONFIGURATION LOGIC: Set False and edit the real-data branch to supply your own prepared DataFrames.
USE_SYNTHETIC = True
OUTPUT = Path('runs/notebook_research')
if USE_SYNTHETIC:
    transactions_df, quotes_df, BASE_FEATURES, config = demo_inputs()
    BASE_CAT_FEATURES = []
    LGB_PARAMS = dict(n_estimators=25, num_leaves=15, learning_rate=.1, n_jobs=2, verbosity=-1)
else:
    # FILE IO LOGIC: These inputs must already contain causal baseline features and compatible spreads.
    transactions_df = pd.read_parquet('data/transactions.parquet')
    quotes_df = pd.read_parquet('data/quotes.parquet')
    # CONFIGURATION LOGIC: Replace illustrative names or edit them in the form before validation.
    BASE_FEATURES = ['previous_spread', 'size', 'remaining_years', 'side']
    BASE_CAT_FEATURES = ['side']
    LGB_PARAMS = dict(n_estimators=200, num_leaves=31, learning_rate=.05, n_jobs=4)
    config = PipelineConfig(
        TransactionColumns(bond='bond_id', time='prediction_time', target='spread', actual='spread',
                           quantity='size', maturity_years='remaining_years', anchor='previous_spread'),
        QuoteColumns(bond='bond_id', known_time='known_time', dealer='dealer_id',
                     bid='bid_spread', ask='ask_spread', quantity='quote_size'),
        timezone='America/New_York', quote_timezone='UTC', target_scale=1., quote_scale=1., error_scale=1.,
        unit='bps', value_kind='spread',
    )
training = TrainingConfig(target_mode='level', validation_fraction=.2, test_fraction=.2,
                          embargo_dates=1, test_embargo_dates=1, min_train_dates=3)

# %% [markdown]
# ## Configure and run
#
# The five essential inputs are the transaction DataFrame, quote DataFrame, ordered
# BASE_FEATURES, explicit BASE_CAT_FEATURES and LightGBM parameter dictionary.
# Optional config/training values prefill the editable form. With no config, map
# columns in the form; column names and source meanings are not inferred.
#
# Supply all upstream feature engineering, sector/proxy metadata, quantity scales
# and benchmark adjustments yourself. Zero/negative spreads, unknown sizes,
# simultaneous candidates and crossings remain diagnostic information.

# %%
# UI LOGIC: Validate is read-only preflight; Run validation starts Steps 1–5 on the applied settings.
controller = research_form(transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES,
                           LGB_PARAMS, config=config, training=training, output=OUTPUT)

# %% [markdown]
# ## Review after the form completes
#
# Run this cell after validation finishes. It reads saved predictions without
# refitting and exposes model selectors, custom slices, two intersecting filters
# and two-column heatmaps. Missing metadata remains unsupported.

# %%
# UI LOGIC: Avoid a premature review call when Run All has only displayed the configuration form.
if controller.run is not None:
    run = controller.run
    display(HTML(f'<p>Generated report: <code>{escape(str(run.report))}</code></p>'))
    panel = run.review(stage='Validation')

# %% [markdown]
# ## Optional custom export
#
# Enable the following switch only after the form completes. Replace the columns
# and bins with your declared analysis. Every original transaction metadata column
# remains available. Exports include complete PNG, exact CSV and HTML evidence.

# %%
# CONFIGURATION LOGIC: Set True only when the indicated metadata is mapped and the desired model pair is fixed.
EXPORT_CUSTOM = False
# REPORTING LOGIC: Reuse saved records; neither slicing nor exporting starts training.
if EXPORT_CUSTOM and controller.run is not None:
    run = controller.run
    comparison = run.compare('Base', run.metadata['selection']['selected'], stage='Validation')
    size = Slice('QUANTITY', [0, 1e6, 5e6, float('inf')], right=False)
    if {'SECTOR', 'QUANTITY'}.issubset(comparison.data):
        review_folder = comparison.export(run.output / 'custom_reviews', interactions=[('SECTOR', size)])

# %% [markdown]
# ## Explicit final test
#
# Review validation first, then enable this cell. Only Base and the frozen candidate
# are evaluated using their original Train fits. The engine cannot establish that
# someone has never inspected the same dates before.
#
# Reopen a saved run with `load_run('runs/notebook_research')`. Regenerate its HTML
# with `python -m raw_quote_engine.cli report runs/notebook_research`.

# %%
# CONFIGURATION LOGIC: Run All leaves final-test evaluation disabled.
RUN_FINAL_TEST = False
# EVALUATION LOGIC: A completed validation run is required before the explicit final-test action.
if RUN_FINAL_TEST:
    if controller.run is None:
        raise ValueError('Complete Run validation in the form first.')
    run = controller.run.finalize_test()
    display(HTML(f'<p>Final report: <code>{escape(str(run.report))}</code></p>'))
