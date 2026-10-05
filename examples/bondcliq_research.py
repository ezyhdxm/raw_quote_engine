# %% [markdown]
# # BondCliQ + data_ig: original baseline settings
#
# Change the two input paths, then run the cells in order. This preset uses the
# existing 14 baseline features, LightGBM parameters and five-date validation/test
# split. `data_ig` must already contain its production-engineered baseline columns.
# See `docs/BONDCLIQ_EXAMPLE.md` for the exact data contract and experiment differences.
#
# Every result comes from the supplied data. This notebook contains no saved results.

# %%
# SETUP LOGIC: Import the ready preset and notebook display helpers without reading data.
from pathlib import Path
from html import escape
from IPython.display import HTML, display
from raw_quote_engine.presets.bondcliq import BASE_FEATURES, LGB_PARAMS, run_bondcliq

# CONFIGURATION LOGIC: Edit these two paths; use Path(r"D:\data\file.parquet") on Windows if needed.
DATA_IG_FILE = Path("data/pipeline/data_ig.parquet")
QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
# CONFIGURATION LOGIC: Optional locations and a CUSIP/SECTOR mapping; defaults need no edits.
OUTPUT = Path("runs/bondcliq")
SECTOR_MAP = None
CACHE_DIR = None

# %% [markdown]
# ## Run Steps 1–5 through validation
#
# Known quote time is UTC converted to New York time. Quotes are restricted to the
# original `data_ig` CUSIPs. Baseline settings are imported from the preset so they
# cannot drift from this example. Model predictions add `PREV_BM_SPREAD` to predicted
# `D_BM_SPREAD`; the observed scoring truth is `BM_SPREAD`, with errors in bps.
#
# This cell fits Base, Quote, Quote+Path and Quote+CrossBond. It saves actual tables,
# full PNGs and a report. Progress shows the current stage; repeating unchanged inputs
# reuses completed stages. Changed inputs/settings require a new output directory.

# %%
# ORCHESTRATION LOGIC: The preset performs source preparation, diagnostics, features and validation.
run = run_bondcliq(
    DATA_IG_FILE, QUOTES_FILE, output=OUTPUT,
    sector_map=SECTOR_MAP, cache_dir=CACHE_DIR,
)
# UI LOGIC: Show the report location without printing the result tables into the notebook.
display(HTML(f"<p>Open the generated report: <code>{escape(str(run.report))}</code></p>"))

# %% [markdown]
# ## Review saved validation predictions
#
# Inspect >=1MM trades, prior-anchor/CPP gaps above 5/10 bps and low historical
# liquidity. Separate maturity below one year. The panel can compare any two saved
# models and custom one-/two-column slices without training again.
#
# These engine quote families differ from the older Quote levels / Reliability /
# Age decay chain. This is a new experiment, not a replay of its historical gains.

# %%
# UI LOGIC: Model and slice controls use the saved common-row validation predictions.
panel = run.review(stage="Validation")

# %% [markdown]
# ## Final test: explicitly opt in after validation review
#
# Leave the switch False for the first run. Set it True only after accepting the
# frozen validation selection, then execute this cell. Only Base and that candidate
# are evaluated. The engine reuses train-only fits; it does not apply the old
# notebook's final refit. Previously inspected dates cannot become unseen again.

# %%
# CONFIGURATION LOGIC: This explicit switch keeps Run All from evaluating test by default.
RUN_FINAL_TEST = False
# EVALUATION LOGIC: Reuse frozen selection and fitted models, then refresh the generated report.
if RUN_FINAL_TEST:
    run = run.finalize_test()
    display(HTML(f"<p>Final report: <code>{escape(str(run.report))}</code></p>"))
