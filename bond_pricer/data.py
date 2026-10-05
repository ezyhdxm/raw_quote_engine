"""Explicit, keyed inputs for paired prediction comparisons. Nothing trains on import."""
# SETUP LOGIC: Keep this package independent of BondCliQ, LightGBM and private loaders.
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def read_data(source):
    # FILE IO LOGIC: Accept an in-memory table or a local CSV/Parquet file; never deserialize models.
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in {'.parquet', '.parque', '.pq'}:
        return pd.read_parquet(path)
    raise ValueError('Use a pandas DataFrame, CSV or Parquet file.')


def check_key(frame, column):
    # VALIDATION LOGIC: A stable key is mandatory for joins; duplicates must be resolved upstream.
    if column not in frame or frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f'{column!r} must exist, contain no missing values and be unique.')


def attach_predictions(data, predictions, *, on, columns=None):
    """Left-join a wide prediction table by identity; missing predictions remain visible."""
    # FILE IO LOGIC: Materialize independent input tables.
    left, right = read_data(data), read_data(predictions)
    check_key(left, on)
    check_key(right, on)
    columns = columns or {name: name for name in right if name != on}
    # VALIDATION LOGIC: Require unambiguous new names; accidental overwrites are errors.
    names = list(columns.values())
    if on in columns or len(set(names)) != len(names) or set(names) & set(left):
        raise ValueError('Prediction names must be unique new columns; do not rename the join key.')
    if not set(columns).issubset(right):
        raise ValueError('A requested prediction column is missing.')
    # CORE LOGIC: STEP 1 — Align predictions by the trade key, preserving the left population.
    # Input: trades id=[2,1,3]; predictions id=[1,2], p=[10,20]; columns={'p':'BASE'}.
    # Output: id=[2,1,3], BASE=[20,10,NaN].
    # Explanation: ID 2 receives 20, ID 1 receives 10, and unmatched ID 3 remains present.
    # Trick: A left one-to-one merge prevents positional misalignment and hidden row multiplication.
    selected = right[[on, *columns]].rename(columns=columns)
    result = left.merge(selected, on=on, how='left', sort=False, validate='one_to_one')
    return result


def from_long_predictions(data, predictions, *, on='row_id', model='model', value='pred_spread',
                          stage_column='stage', stage=None):
    """Convert saved model/row predictions to wide columns without opening other stages."""
    # FILE IO LOGIC: Read only the explicit table supplied by the caller.
    predictions = read_data(predictions)
    # VALIDATION LOGIC: Multiple stages require an explicit choice; never silently blend holdouts.
    required = {on, model, value}
    if not required.issubset(predictions):
        raise ValueError(f'Prediction table needs {sorted(required)}.')
    if stage_column in predictions:
        stages = predictions[stage_column].dropna().unique()
        if stage is None and (len(stages) > 1 or predictions[stage_column].isna().any()):
            raise ValueError('Select one stage explicitly before comparing these predictions.')
    elif stage is not None:
        raise ValueError(f'Cannot select a stage without {stage_column!r}.')
    # CORE LOGIC: STEP 1 — Select a stage and pivot unique model/trade observations.
    # Input: (id,model,stage,p)=[(1,'A','Test',10),(1,'B','Test',11),(1,'A','Train',9)], stage='Test'.
    # Output: one wide row (id=1,A=10,B=11).
    # Explanation: Train is excluded before pivoting; A and B stay paired on trade 1.
    # Trick: Duplicate model/key observations raise instead of being averaged.
    selected = predictions if stage is None else predictions.loc[predictions[stage_column].eq(stage)]
    if selected.empty or selected[[on, model]].isna().any().any():
        raise ValueError('Selected predictions are empty or have missing model/trade keys.')
    if selected.duplicated([on, model]).any():
        raise ValueError('Duplicate model/trade predictions; select one run and stage first.')
    wide = selected.pivot(index=on, columns=model, values=value).reset_index()
    return attach_predictions(data, wide, on=on)


def numeric(frame, column):
    # VALIDATION LOGIC: Coerce only the specified numerical field; retain the source table.
    if column not in frame:
        raise ValueError(f'Missing column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def local_time(values, timezone):
    # TIME CONVERSION LOGIC: Naive timestamps are local; ambiguous/nonexistent DST times are unassigned.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='In a future version of pandas, parsing datetimes with mixed time zones',category=FutureWarning)
            result = pd.to_datetime(values, errors='coerce', format='mixed')
        if result.dt.tz is None:
            return result.dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
        return result.dt.tz_convert(timezone)
    except (AttributeError, ValueError) as exc:
        # TIME CONVERSION LOGIC: Mixed explicit UTC offsets across DST are still unambiguous instants.
        parsed = values.map(lambda value: pd.to_datetime(value,errors='coerce'))
        available = parsed.dropna()
        if not available.empty and available.map(lambda value: value.tzinfo is not None).all():
            return pd.to_datetime(values,errors='coerce',format='mixed',utc=True).dt.tz_convert(timezone)
        raise ValueError('Use uniformly naive local timestamps or explicitly timezone-aware timestamps; do not mix them.') from exc


def paired_rows(data, actual, reference, candidate, *, id_column=None, time_column=None,
                bond_column=None, error_scale=1.0, timezone='America/New_York',
                reference_anchor=None, candidate_anchor=None):
    """Keep the common finite sample, recording every exclusion and side's own coverage."""
    # VALIDATION LOGIC: Error units and identity are explicit, never inferred from observed errors.
    frame = data if isinstance(data,pd.DataFrame) else read_data(data)
    if frame.columns.duplicated().any() or any(str(c).startswith('__') for c in frame):
        raise ValueError('Input columns must be unique; names starting with __ are reserved.')
    if not np.isfinite(error_scale) or error_scale <= 0:
        raise ValueError('error_scale must be a positive finite number.')
    if id_column is not None:
        check_key(frame, id_column)
    if reference == candidate or actual in {reference, candidate}:
        raise ValueError('Choose two different prediction columns, separate from the actual target.')
    # CORE LOGIC: STEP 1 — Convert target and predictions, optionally adding their own anchors.
    # Input: actual=[1.02,1.03], reference=[.01,.02], anchor=[1,1], candidate=[1.02,1.04].
    # Output: truth=[1.02,1.03], a=[1.01,1.02], b=[1.02,1.04].
    # Explanation: Only the delta reference receives its configured anchor; candidate is already a level.
    # Trick: All three resulting values must share units; error_scale is applied once, after subtraction.
    truth = numeric(frame, actual)
    a, b = numeric(frame, reference), numeric(frame, candidate)
    if reference_anchor is not None:
        a = a + numeric(frame, reference_anchor)
    if candidate_anchor is not None:
        b = b + numeric(frame, candidate_anchor)
    # CORE LOGIC: STEP 2 — Form a single common finite mask for the comparison.
    # Input: truth=[10,10,10,NaN], a=[11,NaN,11,11], b=[10,9,NaN,10].
    # Output: target_ok=[True,True,True,False], paired=[True,False,False,False].
    # Explanation: Row 1 alone has all three finite values; neither model gets a different denominator.
    # Trick: Missing predictions and overflowed differences are coverage gaps, never filled with zero.
    target_ok = np.isfinite(truth)
    a_ok, b_ok = np.isfinite(a), np.isfinite(b)
    with np.errstate(over='ignore', invalid='ignore'):
        error_a, error_b = (a-truth)*error_scale, (b-truth)*error_scale
    finite_errors = np.isfinite(error_a) & np.isfinite(error_b)
    paired = target_ok & a_ok & b_ok & finite_errors
    rows = frame.loc[paired].copy().reset_index(drop=True)
    # CORE LOGIC: STEP 3 — Materialize signed and absolute errors on those same rows.
    # Input: truth=[1], a=[1.02], b=[.99], error_scale=100.
    # Output: reference_error≈[2], candidate_error≈[-1], absolute_errors≈[2],[1] (floating point).
    # Explanation: (prediction−actual)×100 converts percentage-point spreads to basis points.
    # Trick: NumPy arrays avoid realigning a retained source index against the new consecutive index.
    rows['__actual'] = truth.loc[paired].to_numpy()
    rows['__reference'], rows['__candidate'] = a.loc[paired].to_numpy(), b.loc[paired].to_numpy()
    rows['__error_reference'] = error_a.loc[paired].to_numpy()
    rows['__error_candidate'] = error_b.loc[paired].to_numpy()
    rows['__ae_reference'] = rows['__error_reference'].abs()
    rows['__ae_candidate'] = rows['__error_candidate'].abs()
    # TIME CONVERSION LOGIC: Missing temporal metadata does not remove otherwise evaluable predictions.
    times = local_time(rows[time_column], timezone) if time_column else pd.Series(pd.NaT, index=rows.index)
    rows['__date'], rows['__hour'] = times.dt.strftime('%Y-%m-%d'), times.dt.hour
    rows['__bond'] = rows[bond_column] if bond_column else pd.Series(pd.NA, index=rows.index)
    # CORE LOGIC: STEP 4 — Report coverage against the complete supplied evaluation population.
    # Input: target_ok=[T,T,T,F], a_ok=[T,F,T,T], b_ok=[T,T,F,T], paired=[T,F,F,F].
    # Output: total=4, valid_target=3, reference_available=2, candidate_available=2, paired=1, excluded=3.
    # Explanation: Per-model availability counts finite predictions only where the target is finite.
    # Trick: Independent missing categories may overlap; only total−paired is an exclusive exclusion count.
    coverage = dict(total=len(frame), valid_target=int(target_ok.sum()), paired=int(paired.sum()))
    coverage.update(reference_available=int((target_ok & a_ok).sum()), candidate_available=int((target_ok & b_ok).sum()))
    coverage.update(excluded=len(frame)-len(rows), missing_time=int(times.isna().sum()))
    coverage['nonfinite_error_rows'] = int((target_ok & a_ok & b_ok & ~finite_errors).sum())
    return rows, coverage
