# %% [markdown]
# # BondCliQ and data_ig: editable research example
#
# Load the prepared transaction table and raw quote file below, then run the cells.
# The last setup cell opens a form: inspect its settings, click **Validate inputs**, then
# **Run validation**. Running all notebook cells alone never starts training.
#
# All dataset preparation lives in this notebook. The engine receives DataFrames,
# explicit feature lists, model parameters and source mappings; it has no dataset preset.
# This baseline uses 15 fields with separate counterparty and side categories. It is
# a revised baseline, so historical BASE14 scores are not reproduced.

# %%
# SETUP LOGIC: Import the generic engine and local notebook display helpers.
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from raw_quote_engine import PipelineConfig, TransactionColumns, QuoteColumns, TrainingConfig, research_form

# CONFIGURATION LOGIC: Change the two paths; prepared data_ig already contains upstream BASE features.
DATA_IG_FILE = Path("data/pipeline/data_ig.parquet")
QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
OUTPUT = Path("runs/bondcliq_separate_categories")
SECTOR_MAP_FILE = None  # Optional CSV or Parquet containing CUSIP and SECTOR.

# FILE IO LOGIC: Read source tables; their files are never overwritten by this example.
data_ig = pd.read_parquet(DATA_IG_FILE)
bcq_df = pd.read_parquet(QUOTES_FILE)

# %%
# CONFIGURATION LOGIC: Separate observed counterparty and side replace the earlier joint category.
BASE_FEATURES = [
    'D_CPP_BM_SPREAD', 'COUPON', 'D_CDX_TRADE', 'MEAN_ISSUER_SPREAD_DEV',
    'NUM_OF_ISSUER_TRADES_SINCE_PREV', 'PREV_BM_SPREAD', 'PREV_QUANTITY',
    'PREV_TRADE_TYPE', 'QUANTITY', 'CONTRA_PARTY_TYPE', 'SIDE', 'YRS_TO_MATURITY',
    'BM_YIELD_STD', 'D_SHORT_TO_BM_SPREAD', 'PREV_BM_SPREAD_STD_GROUP_BY_TYPE',
]
BASE_CAT_FEATURES = ['PREV_TRADE_TYPE', 'CONTRA_PARTY_TYPE', 'SIDE']
LGB_PARAMS = dict(
    objective='mae', boosting_type='dart', n_estimators=400, learning_rate=.2,
    num_leaves=127, max_bin=511, max_depth=-1, min_child_samples=20,
    min_split_gain=0., subsample=1., subsample_freq=0, colsample_bytree=1.,
    reg_alpha=0., reg_lambda=0., n_jobs=8, verbosity=-1, random_state=2026,
)
# VALIDATION LOGIC: Missing engineered inputs require upstream preparation rather than invented substitutes.
required = BASE_FEATURES + ['CUSIP', 'ISSUER', 'EFFECTIVE_DATETIME_TS', 'D_BM_SPREAD', 'BM_SPREAD']
missing = sorted(set(required) - set(data_ig))
if missing:
    raise ValueError(f"Prepared data_ig is missing: {missing}")

# %% [markdown]
# ## Optional user-owned sector mapping
#
# Leave `SECTOR_MAP_FILE=None` to retain existing metadata. If supplied, this map
# overrides matching CUSIPs only. It creates no bonds and changes no BASE features.
# Review sector definitions and their availability upstream; the engine does not
# infer an issuer/sector map or retrieve a mapping file.

# %%
# FILE IO LOGIC: Optional metadata is read here, outside the generic engine.
sector_map = None
if SECTOR_MAP_FILE is not None:
    source = Path(SECTOR_MAP_FILE)
    sector_map = pd.read_csv(source, dtype='string') if source.suffix.lower() == '.csv' else pd.read_parquet(source)
# CORE LOGIC: STEP 1 — Normalize identifiers and freeze the full supplied transaction-bond universe.
# Input: data_ig CUSIP=[' X ','Y',None], original indexes=[7,7,9].
# Output: model_data CUSIP=['X','Y',<NA>], row_id=[0,1,2], quote_universe.cusip=['X','Y'].
# Explanation: Universe membership precedes label filtering, so a missing target cannot erase a quoted bond.
# Trick: Resetting row indexes gives distinct transactions stable positional identities for this input order.
model_data = data_ig.reset_index(drop=True).copy()
model_data['row_id'] = np.arange(len(model_data), dtype='int64')
model_data['CUSIP'] = model_data.CUSIP.astype('string').str.strip().replace('', pd.NA)
quote_universe = pd.DataFrame({'cusip': model_data.CUSIP.dropna().drop_duplicates().to_numpy()})
# CORE LOGIC: STEP 2 — Apply only explicit nonblank, unambiguous sector corrections.
# Input: trades=[('X','Unknown'),('Y','TMT')], map=[('X','Energy'),('X','Energy')].
# Output: trades=[('X','Energy'),('Y','TMT')]; repeated identical map rows collapse.
# Explanation: A left-side lookup preserves transactions; conflicting sectors for one CUSIP raise.
# Trick: Unmatched rows retain their existing sector, and unknown sectors remain visible.
if sector_map is not None:
    mapping = sector_map[['CUSIP', 'SECTOR']].astype('string').apply(lambda s: s.str.strip())
    mapping = mapping.replace('', pd.NA).dropna().drop_duplicates()
    if mapping.CUSIP.duplicated().any():
        raise ValueError('Conflicting SECTOR labels in the supplied CUSIP map.')
    previous = model_data.get('SECTOR', pd.Series(pd.NA, index=model_data.index, dtype='string'))
    model_data['SECTOR'] = model_data.CUSIP.map(mapping.set_index('CUSIP').SECTOR).fillna(previous)

# %% [markdown]
# ## Explicit eligibility and experiment window
#
# The supplied `data_ig` defines the intended three-month universe. This example
# retains finite target/anchor/truth rows with valid keys through the final local
# quote date. No-quote transactions remain. The engine uses the last five observed
# dates for Test, the previous five for Validation, and two embargo dates before
# Validation. At least ten earlier training dates are required.

# %%
# CORE LOGIC: STEP 1 — Normalize supplied model numerics and retain separate source categories.
# Input: COUPON=['4.5','bad'], PREV_BM_SPREAD=[1,<NA>], SIDE=['B','S'].
# Output: COUPON=[4.5,NaN], PREV_BM_SPREAD=[1.0,NaN], SIDE=['B','S'].
# Explanation: Numeric coercion does not rebuild production features or reinterpret trade direction.
# Trick: float64 turns nullable missing values into NaN before the finite-row mask.
for column in set(BASE_FEATURES + ['D_BM_SPREAD', 'BM_SPREAD']) - set(BASE_CAT_FEATURES):
    model_data[column] = pd.to_numeric(model_data[column], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
# TIME NORMALIZATION LOGIC: Prepared naive trade times mean New York; raw naive quote times mean UTC.
trade_time = pd.to_datetime(model_data.EFFECTIVE_DATETIME_TS, errors='coerce', format='mixed')
trade_time = trade_time.dt.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise') if trade_time.dt.tz is None else trade_time.dt.tz_convert('America/New_York')
model_data['EFFECTIVE_DATETIME_TS'] = trade_time
quote_time = pd.to_datetime(bcq_df.quote_timestamp_UTC, errors='coerce', utc=True, format='mixed')
quote_cusip = bcq_df.cusip.astype('string').str.strip()
quote_end = quote_time.loc[quote_cusip.isin(quote_universe.cusip)].max()
if pd.isna(quote_end):
    raise ValueError('No valid quote timestamp in the supplied transaction-bond universe.')
# CORE LOGIC: STEP 2 — Apply one common label/key mask and cap at the final local quote date.
# Input: row_id=[0,1,2], finite labels=[True,False,True], valid keys all True, within date=[True,True,False].
# Output: retained row_id=[0], excluded_rows=2; row 0 survives even without any quote.
# Explanation: The cutoff includes the full last quote date, not only times before its final message.
# Trick: Missing target/anchor/truth rows are excluded once for all models, while quote_universe stays unchanged.
finite = np.isfinite(model_data[['D_BM_SPREAD', 'PREV_BM_SPREAD', 'BM_SPREAD']]).all(axis=1)
valid_keys = model_data[['CUSIP', 'EFFECTIVE_DATETIME_TS']].notna().all(axis=1)
within = trade_time.dt.normalize().le(quote_end.tz_convert('America/New_York').normalize())
eligible = finite & valid_keys & within
excluded_rows = int((~eligible).sum())
model_data = model_data.loc[eligible].copy()

# %%
# CONFIGURATION LOGIC: Optional metadata mappings use only explicitly present source fields; no proxy is built.
transactions = TransactionColumns(
    bond='CUSIP', time='EFFECTIVE_DATETIME_TS', target='D_BM_SPREAD', actual='BM_SPREAD', id='row_id',
    issuer='ISSUER', sector='SECTOR' if 'SECTOR' in model_data else None,
    quantity='QUANTITY', prev_quantity='PREV_QUANTITY', maturity_years='YRS_TO_MATURITY',
    anchor='PREV_BM_SPREAD', cpp='MID_SPREAD_CPP' if 'MID_SPREAD_CPP' in model_data else None,
    rollover_adjustment='D_BM_YIELD_OFFSET' if 'D_BM_YIELD_OFFSET' in model_data else None,
)
config = PipelineConfig(
    transactions, QuoteColumns(bond='cusip', known_time='quote_timestamp_UTC', dealer='firm',
                               side='side', value='spread', quantity='quantity'),
    timezone='America/New_York', quote_timezone='UTC', value_kind='spread',
    target_scale=1., quote_scale=.01, error_scale=100., unit='bps', clip_floor=.10,
    age_min=30., sync_min=1., lookback_min=30., allow_exact=True,
    quantity_scale=1., quote_quantity_scale=1.,
    priority_history_column='TRADE_COUNTS_PREV_MONTH' if 'TRADE_COUNTS_PREV_MONTH' in model_data else None,
)
training = TrainingConfig(
    validation_dates=5, test_dates=5, embargo_dates=2, test_embargo_dates=0, min_train_dates=10,
    target_mode='delta', category_order='appearance', apply_model_defaults=False, selection_slice='large_long',
)
# UI LOGIC: Opening the form does not train; inspect mappings and click Validate, then Run validation.
controller = research_form(model_data, bcq_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
                           config=config, training=training, output=OUTPUT, quote_universe=quote_universe)

# %% [markdown]
# ## Target reconstruction and review
#
# The target stays exactly as supplied. This example expects the upstream label
# convention `D_BM_SPREAD = BM_SPREAD - PREV_BM_SPREAD + D_BM_YIELD_OFFSET` when
# the optional offset is mapped. Predictions reconstruct as
# `predicted_delta + PREV_BM_SPREAD - D_BM_YIELD_OFFSET`; the explicit scoring
# truth is `BM_SPREAD`. Without an adjustment mapping its value is zero.
#
# A mapped `MID_SPREAD_CPP` is compared with `PREV_BM_SPREAD - adjustment` for
# anchor-quality slices. No CPP proxy or rowwise fallback is synthesized.
# Verify upstream benchmark and unit conventions before running. The scalar
# adjustment does not repair cross-time quote benchmark changes.
#
# Once the form finishes, run the cell below to inspect saved validation results.
# Review >=1MM, weak-liquidity and >5/10-bps anchor-gap groups; separate <1y maturity.

# %%
# UI LOGIC: Review becomes available after the explicit form action completes.
if controller.run is not None:
    run = controller.run
    display(HTML(f'<p>Generated report: <code>{escape(str(run.report))}</code></p>'))
    panel = run.review(stage='Validation')

# %% [markdown]
# ## Explicit final test
#
# Leave the switch False until validation is reviewed. Test uses the same
# Train-fitted Base and the frozen selected candidate, without historical refitting.
# The revised BASE15 and engine quote families form a new experiment; saved old
# BASE14 gains are not comparable evidence. All results are computed from this run.

# %%
# CONFIGURATION LOGIC: Run All keeps final-test evaluation off.
RUN_FINAL_TEST = False
# EVALUATION LOGIC: Require a completed validation run before explicitly opening final test.
if RUN_FINAL_TEST:
    if controller.run is None:
        raise ValueError('Complete Run validation in the form first.')
    run = controller.run.finalize_test()
    display(HTML(f'<p>Final report: <code>{escape(str(run.report))}</code></p>'))
