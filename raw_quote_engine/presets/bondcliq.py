"""BondCliQ / prepared data_ig preset matching the research baseline and validation split."""
# SETUP LOGIC: Importing this adapter does not read files, build features, or train models.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import PipelineConfig, QuoteColumns, TransactionColumns
from ..ingest import read_frame, _times
from ..training import TrainingConfig
from ..pipeline import run_research, render
from ..cache import Progress, write_json

# CONFIGURATION LOGIC: Keep the research BASE14 order, target units and explicit estimator parameters.
BASE_FEATURES = ['D_CPP_BM_SPREAD', 'COUPON', 'D_CDX_TRADE', 'MEAN_ISSUER_SPREAD_DEV',
                 'NUM_OF_ISSUER_TRADES_SINCE_PREV', 'PREV_BM_SPREAD', 'PREV_QUANTITY',
                 'PREV_TRADE_TYPE', 'QUANTITY', 'CONTRA_PARTY_SIDE', 'YRS_TO_MATURITY',
                 'BM_YIELD_STD', 'D_SHORT_TO_BM_SPREAD', 'PREV_BM_SPREAD_STD_GROUP_BY_TYPE']
BASE_CAT_FEATURES = ['PREV_TRADE_TYPE', 'CONTRA_PARTY_SIDE']
LGB_PARAMS = dict(objective='mae', boosting_type='dart', n_estimators=400, learning_rate=.2,
                  num_leaves=127, max_bin=511, max_depth=-1, min_child_samples=20,
                  min_split_gain=0., subsample=1., subsample_freq=0, colsample_bytree=1.,
                  reg_alpha=0., reg_lambda=0., n_jobs=8, verbosity=-1, random_state=2026)


def joint_context(frame):
    # CORE LOGIC: STEP 1 — Encode unique observed counterparty/side pairs without interpreting direction.
    # Input: pairs=[['C','B'],['D',null],['C','B']].
    # Output: unique=[['C','B'],['D',null]], labels=['["C","B"]','["D",null]'].
    # Explanation: Each distinct pair is encoded once as a JSON array; null stays distinct from 'null'.
    # Trick: No trimming, uppercasing or unreliable TRADE_TYPE lookup changes these raw source categories.
    pairs = frame[['CONTRA_PARTY_TYPE', 'SIDE']].astype('string')
    unique = pairs.drop_duplicates()
    labels = [json.dumps([None if pd.isna(v) else str(v) for v in values],
                         ensure_ascii=False, separators=(',', ':'))
              for values in unique.itertuples(index=False, name=None)]
    # CORE LOGIC: STEP 2 — Expand the unique labels back to the original row order.
    # Input: pairs=[['C','B'],['D',null],['C','B']], unique labels=['["C","B"]','["D",null]'].
    # Output: positions=[0,1,0], result=['["C","B"]','["D",null]','["C","B"]'].
    # Explanation: MultiIndex matches both components; repeated source index labels cannot reorder records.
    # Trick: Return an array so assignment is positional, not a pandas index-alignment operation.
    positions = pd.MultiIndex.from_frame(unique).get_indexer(pd.MultiIndex.from_frame(pairs))
    return np.asarray(labels, dtype=object)[positions]


def add_sector_map(frame, sector_map):
    # FILE IO LOGIC: A user-supplied CUSIP/SECTOR map enriches metadata only; no external lookup is fetched.
    if sector_map is None:
        return frame
    mapping = read_frame(sector_map, ['CUSIP', 'SECTOR'])
    if not {'CUSIP', 'SECTOR'}.issubset(mapping):
        raise ValueError('sector_map must contain CUSIP and SECTOR columns.')
    # CORE LOGIC: STEP 1 — Validate one nonblank sector per mapped CUSIP.
    # Input: map=[('X','Energy'),('X','Energy'),('Y',null)].
    # Output: mapping index={'X':'Energy'}; duplicate identical labels collapse, missing labels do not fill.
    # Explanation: Strip metadata whitespace, then remove empty keys/values and exact duplicate pairs.
    # Trick: Conflicting nonmissing sectors for one CUSIP raise instead of choosing the first row.
    mapping = mapping[['CUSIP', 'SECTOR']].astype('string').apply(lambda s: s.str.strip())
    mapping = mapping.replace('', pd.NA).dropna().drop_duplicates()
    if mapping.CUSIP.duplicated().any():
        raise ValueError('sector_map has conflicting SECTOR values for the same CUSIP.')
    lookup = mapping.set_index('CUSIP').SECTOR
    # CORE LOGIC: STEP 2 — Apply explicit metadata corrections only to supplied transaction records.
    # Input: trades=[('X','Unknown'),('Z','TMT')], lookup={'X':'Energy','Y':'BankFin'}.
    # Output: [('X','Energy'),('Z','TMT')]; no Y transaction or quote-universe member is added.
    # Explanation: A mapped sector overrides the source label; unmatched rows keep their existing sector.
    # Trick: The left population is unchanged, and no model feature is recomputed from this descriptive map.
    result = frame.copy()
    original = result.get('SECTOR', pd.Series(pd.NA, index=result.index, dtype='string'))
    result['SECTOR'] = result.CUSIP.astype('string').str.strip().map(lookup).fillna(original)
    return result


def cpp_proxy(frame):
    # CONFIGURATION LOGIC: Match the historical audit's one-source-per-schema precedence; never fill rowwise.
    if {'MID_SPREAD_CPP', 'D_BM_YIELD_OFFSET'}.issubset(frame):
        source = 'midpoint_with_offset'
    elif {'D_CPP_BM_SPREAD_BID', 'D_CPP_BM_SPREAD_ASK'}.issubset(frame):
        source = 'mean_bid_ask_delta'
    else:
        source = 'side_dependent_delta'
    # CORE LOGIC: STEP 1 — Build a diagnostic aligned CPP level without changing BASE14.
    # Input: PREV=[1,1], MID=[1.1,NaN], OFFSET=[.02,.02], source='midpoint_with_offset'.
    # Output: CPP_AUDIT_LEVEL=[1.12,NaN]; absolute anchor discrepancies≈[12,NaN] bps.
    # Explanation: Midpoint plus benchmark-roll offset aligns the CPP level to the legacy audit convention.
    # Trick: Other sources produce anchor+delta; missing selected-source values never borrow another field.
    number = lambda name: pd.to_numeric(frame[name], errors='coerce').astype(float)
    if source == 'midpoint_with_offset':
        level = number('MID_SPREAD_CPP') + number('D_BM_YIELD_OFFSET')
    elif source == 'mean_bid_ask_delta':
        level = number('PREV_BM_SPREAD') + (number('D_CPP_BM_SPREAD_BID') + number('D_CPP_BM_SPREAD_ASK')) / 2
    else:
        level = number('PREV_BM_SPREAD') + number('D_CPP_BM_SPREAD')
    return level.replace([np.inf, -np.inf], np.nan), source


def cpp_discrepancy(frame, source):
    # CORE LOGIC: STEP 1 — Preserve the historical arithmetic for strict 5/10-bps cohort boundaries.
    # Input: source='side_dependent_delta', D_CPP_BM_SPREAD=[.05,.10,.1001], PREV=[1,1,1].
    # Output: CPP_AUDIT_GAP_BPS=[5,10,10.01] (last approximately); >5=[False,True,True], >10=[False,False,True].
    # Explanation: Convert the original delta directly, avoiding anchor+delta-anchor cancellation rounding.
    # Trick: The chosen schema remains fixed; midpoint uses MID-PREV+OFFSET in the historical operation order.
    number = lambda name: pd.to_numeric(frame[name], errors='coerce').astype(float)
    if source == 'midpoint_with_offset':
        delta = number('MID_SPREAD_CPP') - number('PREV_BM_SPREAD') + number('D_BM_YIELD_OFFSET')
    elif source == 'mean_bid_ask_delta':
        delta = (number('D_CPP_BM_SPREAD_BID') + number('D_CPP_BM_SPREAD_ASK')) / 2
    else:
        delta = number('D_CPP_BM_SPREAD')
    return (delta.abs() * 100).replace([np.inf, -np.inf], np.nan)


def pilot_splits(frame, quote_end):
    # CORE LOGIC: STEP 1 — Reserve the historical 5/5 observed-date evaluation windows and Train embargo.
    # Input: one trade each weekday 2026-03-02..03-31 (22 dates), quote_end=2026-03-31 10:00 ET.
    # Output: validation dates=03-18..03-24; test dates=03-25..03-31; train dates=03-02..03-13.
    # Explanation: Last five dates are Test, preceding five Validation, preceding two are Train embargo.
    # Trick: These are observed transaction dates through quote end, not calendar days or row fractions.
    days = frame.EFFECTIVE_DATETIME_TS.dt.normalize()
    dates = pd.DatetimeIndex(days.unique()).sort_values()
    dates = dates[dates <= quote_end.normalize()]
    validation_start, test_start = len(dates) - 10, len(dates) - 5
    if validation_start - 2 < 10:
        raise ValueError(f'BondCliQ pilot needs at least 22 observed trade dates through quote end; found {len(dates)}.')
    # CORE LOGIC: STEP 2 — Label original rows; the two Train embargo dates remain available as history.
    # Input: query dates=[2026-03-13,03-16,03-18,03-25], boundaries from the example above.
    # Output: ['Train','Embargo','Validation','Test'] in the original row order.
    # Explanation: No extra embargo is inserted between Validation and Test for this historical split.
    # Trick: The engine's separate final-test operation reuses fitted Train models, unlike historical refitting.
    labels = pd.Series('Embargo', index=frame.index)
    labels.loc[days.isin(dates[:validation_start - 2])] = 'Train'
    labels.loc[days.isin(dates[validation_start:test_start])] = 'Validation'
    labels.loc[days.isin(dates[test_start:])] = 'Test'
    return labels


def preset_config():
    # CONFIGURATION LOGIC: Preserve percent-point training labels; only quotes convert bps to percent points.
    transactions = TransactionColumns(bond='CUSIP', time='EFFECTIVE_DATETIME_TS', target='D_BM_SPREAD',
        actual='BM_SPREAD', id='row_id', issuer='ISSUER', sector='SECTOR', quantity='QUANTITY',
        prev_quantity='PREV_QUANTITY', maturity_years='YRS_TO_MATURITY', anchor='PREV_BM_SPREAD',
        cpp='CPP_AUDIT_LEVEL', split='split')
    quotes = QuoteColumns(bond='cusip', known_time='quote_timestamp_UTC', dealer='firm',
                          side='side', value='spread', quantity='quantity')
    return PipelineConfig(transactions, quotes, target_scale=1., quote_scale=.01, error_scale=100.,
        unit='bps', value_kind='spread', timezone='America/New_York', age_min=30., sync_min=1.,
        lookback_min=30., clip_floor=.10, allow_exact=True, quantity_scale=1., quote_quantity_scale=1.,
        priority_history_column='TRADE_COUNTS_PREV_MONTH', priority_cpp_gap_column='CPP_AUDIT_GAP_BPS')


def prepare_bondcliq_inputs(data_ig, quotes, *, sector_map=None):
    """Return prepared transactions, quotes, config, training, full universe, and auditable preparation counts."""
    # FILE IO LOGIC: Read source frames without mutating them or loading the original quote project.
    frame = read_frame(data_ig, ['CUSIP', 'ISSUER', 'SECTOR']).reset_index(drop=True)
    raw = read_frame(quotes, ['cusip', 'firm']).reset_index(drop=True)
    required = set(BASE_FEATURES) - {'CONTRA_PARTY_SIDE'}
    required.update(['CUSIP', 'ISSUER', 'EFFECTIVE_DATETIME_TS', 'D_BM_SPREAD', 'BM_SPREAD', 'CONTRA_PARTY_TYPE', 'SIDE'])
    missing = sorted(required - set(frame))
    quote_missing = sorted({'cusip', 'firm', 'side', 'spread', 'quantity', 'quote_timestamp_UTC'} - set(raw))
    if missing or quote_missing:
        raise ValueError(f'Missing data_ig columns: {missing}; missing BondCliQ columns: {quote_missing}. Use the prepared data_ig with original BASE fields.')
    # VALIDATION LOGIC: Source raw frames must not contain outputs from an earlier engine feature run.
    if {'CPP_AUDIT_LEVEL', 'CPP_AUDIT_GAP_BPS'} & set(frame):
        raise ValueError('CPP_AUDIT fields are reserved for this preset; pass the original prepared data_ig.')
    # CORE LOGIC: STEP 1 — Establish source row identity and the entire supplied traded-bond universe.
    # Input: CUSIP=[' X ','Y',null], original row indexes=[7,7,9].
    # Output: row_id=[0,1,2], CUSIP=['X','Y',<NA>], universe.cusip=['X','Y'].
    # Explanation: The universe precedes label eligibility and quote-end filtering, retaining all supplied traded bonds.
    # Trick: This does not restrict data_ig to bonds quoted or traded during the shorter quote-file period.
    frame['row_id'] = np.arange(len(frame), dtype='int64')
    frame['CUSIP'] = frame.CUSIP.astype('string').str.strip().replace('', pd.NA)
    universe = pd.DataFrame({'cusip': frame.CUSIP.dropna().drop_duplicates().to_numpy()})
    raw['cusip'] = raw.cusip.astype('string').str.strip().replace('', pd.NA)
    outside_universe = ~raw.cusip.isin(universe.cusip)
    raw = raw.loc[~outside_universe].copy()
    # TIME NORMALIZATION LOGIC: Naive quote timestamps explicitly mean UTC; naive transaction timestamps mean ET.
    raw['quote_timestamp_UTC'] = pd.to_datetime(raw.quote_timestamp_UTC, utc=True, errors='coerce', format='mixed')
    # TIME NORMALIZATION LOGIC: Discard the old notebook's derived ET field; the shared engine recreates it from UTC.
    raw = raw.drop(columns=['quote_timestamp_ET'], errors='ignore')
    frame['EFFECTIVE_DATETIME_TS'] = _times(frame.EFFECTIVE_DATETIME_TS, 'America/New_York', 'data_ig time')
    quote_end = raw.quote_timestamp_UTC.max()
    if pd.isna(quote_end):
        raise ValueError('No valid quote timestamp remains in the original data_ig CUSIP universe.')
    quote_end = quote_end.tz_convert('America/New_York')
    # CORE LOGIC: STEP 2 — Rebuild only the current trade category and normalize model numeric inputs.
    # Input: counterparty='C',side='B',PREV_TRADE_TYPE='S',COUPON='4.5',D_CPP_BM_SPREAD='bad'.
    # Output: CONTRA_PARTY_SIDE='["C","B"]',PREV_TRADE_TYPE='S',COUPON=4.5,D_CPP_BM_SPREAD=NaN.
    # Explanation: Numeric strings become numeric; the previous trade category keeps its source meaning.
    # Trick: Neither unreliable TRADE_TYPE nor recomputed upstream BASE features substitutes for supplied values.
    frame['CONTRA_PARTY_SIDE'] = joint_context(frame)
    for name in set(BASE_FEATURES + ['D_BM_SPREAD', 'BM_SPREAD']) - set(BASE_CAT_FEATURES):
        frame[name] = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
    for name in BASE_CAT_FEATURES:
        frame[name] = frame[name].astype('string')
    frame['CPP_AUDIT_LEVEL'], cpp_source = cpp_proxy(frame)
    frame['CPP_AUDIT_GAP_BPS'] = cpp_discrepancy(frame, cpp_source)
    # CORE LOGIC: STEP 3 — Match the historical single finite-label/key eligibility mask and quote-end cap.
    # Input: rows 0/1/2, finite target/anchor/truth=[True,False,True], valid keys all True, within end=[True,True,False].
    # Output: retain row_id=[0]; invalid_target_or_key_rows=1, after_quote_end_rows=1.
    # Explanation: A valid no-quote transaction survives; source order and IDs do not change after filtering.
    # Trick: The cutoff is end of the final local quote DATE, not the last intraday quote timestamp.
    eligible = np.isfinite(frame[['D_BM_SPREAD', 'PREV_BM_SPREAD', 'BM_SPREAD']]).all(axis=1)
    eligible &= frame[['CUSIP', 'EFFECTIVE_DATETIME_TS']].notna().all(axis=1)
    within = frame.EFFECTIVE_DATETIME_TS.dt.normalize().le(quote_end.normalize())
    excluded, after_end = int((~eligible).sum()), int((eligible & ~within).sum())
    frame = frame.loc[eligible & within].copy()
    frame['split'] = pilot_splits(frame, quote_end)
    # METADATA LOGIC: Sector is diagnostic metadata; missing values remain visible unless an explicit map is supplied.
    frame = add_sector_map(frame, sector_map)
    if 'SECTOR' not in frame:
        frame['SECTOR'] = pd.Series(pd.NA, index=frame.index, dtype='string')
    # CONFIGURATION LOGIC: Explicit splits override generic fractions; original estimator kwargs remain exact.
    config = preset_config()
    training = TrainingConfig(target_mode='delta', embargo_dates=2, min_train_dates=10,
        category_order='appearance', apply_model_defaults=False, selection_slice='large_long')
    # PROVENANCE LOGIC: This receipt describes preparation, not predictive gain or guaranteed market completeness.
    audit = dict(preset='bondcliq_data_ig_v1', source_rows=len(eligible), eligible_rows=len(frame),
        invalid_target_or_key_rows=excluded, after_quote_end_rows=after_end,
        supplied_universe_bonds=len(universe), outside_universe_quote_rows=int(outside_universe.sum()),
        quote_end=str(quote_end), cpp_source=cpp_source,
        prior_count_source='TRADE_COUNTS_PREV_MONTH', prior_count_available='TRADE_COUNTS_PREV_MONTH' in frame,
        sector_map_applied=sector_map is not None, sector_missing_rows=int(frame.SECTOR.isna().sum()),
        validation_days=5, test_days=5, train_embargo_days=2, minimum_train_days=10,
        base_features=BASE_FEATURES, categorical_features=BASE_CAT_FEATURES, model_params=LGB_PARAMS,
        target='D_BM_SPREAD', prediction_anchor='PREV_BM_SPREAD', scoring_truth='BM_SPREAD',
        quote_quantity_units='unknown source units; no par conversion or executable-size claim',
        differences=['Engine candidate families are Quote, Quote+Path, Quote+CrossBond; not historical feature ablations.',
                     'Final Test reuses Train-fitted models; historical final-refit results are not reproduced.',
                     'Supplied data_ig defines the full universe; the caller must supply the intended three-month snapshot.'])
    return frame, raw, config, training, universe, audit


def run_bondcliq(data_ig, quotes, *, output='runs/bondcliq', cache_dir=None, sector_map=None, progress=None):
    """Prepare the fixed research baseline and run Steps 1–5 through validation; no automatic final test."""
    # ORCHESTRATION LOGIC: Fixed preset requires only the existing prepared transactions and raw quote source.
    notify = Progress(progress)
    notify('preflight', 0, 1, 'Loading data_ig and BondCliQ; checking fields, units and fixed pilot dates')
    frame, raw, config, training, universe, audit = prepare_bondcliq_inputs(data_ig, quotes, sector_map=sector_map)
    notify('preflight', 1, 1, f'{len(frame):,} eligible transactions; {len(universe):,} original universe bonds')
    output = Path(output).expanduser().resolve()
    # FILE IO LOGIC: Save preflight beside the run folder before expensive work; do not make its folder nonempty.
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_name(output.name + '_preflight.json'), audit)
    run = run_research(frame, raw, list(BASE_FEATURES), dict(LGB_PARAMS), config, output=output,
        training=training, cache_dir=cache_dir, progress=progress, quote_universe=universe)
    # PROVENANCE LOGIC: Include the source-specific audit in the generated report and saved run, without manual losses.
    run.metadata['source_preset'] = audit
    write_json(run.output / 'manifest.json', run.metadata)
    write_json(run.output / 'bondcliq_preflight.json', audit)
    render(run, config)
    return run
