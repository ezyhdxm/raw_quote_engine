"""Normalize supplied trades and quotes without training, raw-data cleaning, or hidden time shifts."""
# SETUP LOGIC: pandas is the only data dependency; imports do not read or write any source.
from dataclasses import asdict
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .config import PipelineConfig


def read_frame(source, string_columns=None):
    """Read a DataFrame, CSV, or Parquet; preserve mapped CSV identifiers such as leading-zero CUSIPs."""
    # FILE IO LOGIC: Materialize an independent frame; do not mutate the caller's data.
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() == '.csv':
            frame = pd.read_csv(path, dtype={name: 'string' for name in string_columns or []})
        elif path.suffix.lower() in {'.parquet', '.parque', '.pq'}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError('Use a pandas DataFrame or a local CSV/Parquet file.')
    if frame.columns.duplicated().any():
        raise ValueError('Source column names must be unique.')
    return frame


def _required(frame, schema):
    # VALIDATION LOGIC: A requested optional field is still mandatory when explicitly configured.
    missing = set(name for name in asdict(schema).values() if name is not None) - set(frame.columns)
    if missing:
        raise ValueError(f'Configured columns are missing: {sorted(missing)}')


def _reserved(frame, mappings):
    # VALIDATION LOGIC: Never overwrite an unrelated baseline feature with a generated canonical field.
    conflicts = [dest for dest, source in mappings.items() if dest in frame and dest != source]
    if conflicts:
        raise ValueError(f'Reserved canonical columns conflict with source columns: {sorted(conflicts)}')


def _identifier(values):
    # CORE LOGIC: STEP 1 — Normalize surrounding whitespace without guessing identifier case or punctuation.
    # Input: bond=[' 001234AB1 ', '', None].
    # Output: ['001234AB1', <NA>, <NA>].
    # Explanation: A blank identifier cannot identify a traded bond, while its leading zeros stay intact.
    # Trick: pandas StringDtype preserves missing values rather than converting them into the string 'nan'.
    result = values.astype('string').str.strip()
    return result.mask(result.eq(''))


def _number(values, scale=1.0):
    # CORE LOGIC: STEP 1 — Coerce a declared numeric field and scale only its finite values.
    # Input: values=['2', 'bad', 0, -1, inf], scale=100.
    # Output: [200.0, NaN, 0.0, -100.0, NaN].
    # Explanation: Zero and negative values survive; malformed, infinite, or overflowed values stay unknown.
    # Trick: Canonical numeric columns use float64/NaN, including nullable inputs; scaling occurs once and missing never becomes zero.
    values = pd.to_numeric(values, errors='coerce')
    with np.errstate(over='ignore', invalid='ignore'):
        result = values * scale
    return result.replace([np.inf, -np.inf], np.nan).astype(float)


def _quantity(frame, column, scale):
    # CORE LOGIC: STEP 1 — Distinguish original missing sizes from zero and malformed/nonpositive sizes.
    # Input: size=[100,0,None,-1,'bad',inf], scale=1000.
    # Output: quantity=[100000,0,NaN,-1000,NaN,NaN], kind=['Positive','Zero','Missing','Other','Other','Other']; raw values unchanged.
    # Explanation: Only a finite positive size is usable for matching; other cases remain separate diagnostics.
    # Trick: Classification uses the original null mask before coercion, so an invalid string is not called missing.
    raw = frame[column] if column is not None else pd.Series(np.nan, index=frame.index)
    number = _number(raw, scale)
    conditions = [mask.fillna(False).to_numpy(dtype=bool) for mask in [raw.isna(), number.gt(0), number.eq(0)]]
    kind = np.select(conditions, ['Missing', 'Positive', 'Zero'], default='Other')
    return number, pd.Series(kind, index=frame.index), raw


def _times(values, timezone, label):
    """Interpret uniformly naive input in timezone; convert uniformly aware input to timezone."""
    # TIME CONVERSION LOGIC: Missing/unparseable times become NaT; ambiguous local DST times raise.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message='In a future version of pandas, parsing datetimes with mixed time zones.*')
            try:
                times = pd.to_datetime(values, errors='coerce', format='mixed')
            except ValueError:
                times = values.map(lambda value: pd.to_datetime(value, errors='coerce'))
        if times.dtype == object:
            known = times.dropna()
            if len(known) and known.map(lambda value: value.utcoffset() is not None).all():
                return pd.to_datetime(values, errors='coerce', format='mixed', utc=True).dt.tz_convert(timezone)
            raise ValueError('Naive and aware timestamps cannot be mixed.')
        if times.dt.tz is None:
            return times.dt.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
        return times.dt.tz_convert(timezone)
    except Exception as exc:
        raise ValueError(f'{label}: use uniform naive local times or aware timestamps; ambiguous/nonexistent DST or mixed zones require explicit upstream resolution.') from exc


def _split(values):
    # CORE LOGIC: STEP 1 — Canonicalize an explicitly supplied evaluation split without assigning one.
    # Input: [' train ', 'VALIDATION', 'Test', 'embargo'].
    # Output: ['Train', 'Validation', 'Test', 'Embargo'].
    # Explanation: Labels identify the user's supplied partition; this adapter never chooses split dates.
    # Trick: Missing or unknown labels raise instead of leaking unassigned observations into training.
    result = _identifier(values).str.lower().map({'train': 'Train', 'validation': 'Validation', 'test': 'Test', 'embargo': 'Embargo'})
    if result.isna().any():
        raise ValueError('Configured split must contain only Train, Validation, Test, or Embargo.')
    return result


def _transaction_mappings(columns):
    # CONFIGURATION LOGIC: Only declared optional fields receive canonical names.
    mappings = {'row_id': columns.id, 'cusip': columns.bond, 'time': columns.time, 'target': columns.target}
    optional = {'ISSUER': columns.issuer, 'SECTOR': columns.sector, 'QUANTITY': columns.quantity,
                'PREV_QUANTITY': columns.prev_quantity, 'MATURITY_YEARS': columns.maturity_years,
                'MATURITY_DATE': columns.maturity_date, 'anchor': columns.anchor, 'cpp': columns.cpp, 'split': columns.split}
    mappings.update({name: source for name, source in optional.items() if source is not None})
    if columns.maturity_date is not None and columns.maturity_years is None:
        mappings['MATURITY_YEARS'] = None
    return mappings


def _maturity_years(frame, columns, times, timezone):
    # CORE LOGIC: STEP 1 — Compute remaining calendar days when an explicit years field was not supplied.
    # Input: trade time='2026-01-01 15:00 America/New_York', maturity_date='2027-01-01'.
    # Output: MATURITY_DATE='2027-01-01 00:00 America/New_York', MATURITY_YEARS=365/365.25≈0.9993155373.
    # Explanation: Calendar dates avoid an intraday fraction or a DST-hour distortion in a dated maturity.
    # Trick: The 365.25-day convention is explicit; supplied maturity_years takes precedence when available.
    dates = _times(frame[columns.maturity_date], timezone, 'maturity_date')
    calendar_dates = dates.dt.tz_localize(None).dt.normalize()
    trade_dates = times.dt.tz_localize(None).dt.normalize()
    years = (calendar_dates - trade_dates).dt.total_seconds() / (86400 * 365.25)
    return dates, years


def normalize_transactions(data, config):
    """Return canonical transactions plus original columns; invalid labels remain NaN and are diagnosed."""
    # VALIDATION LOGIC: Identity and time define a trade; unavailable target labels must not erase its history.
    if not isinstance(config, PipelineConfig):
        raise TypeError('config must be a PipelineConfig.')
    columns = config.transactions
    identifiers = [name for name in [columns.bond, columns.id, columns.issuer, columns.sector] if name]
    original = read_frame(data, identifiers).reset_index(drop=True)
    frame = original.copy()
    _required(frame, columns)
    _reserved(frame, _transaction_mappings(columns))
    # CORE LOGIC: STEP 1 — Establish stable transaction keys and the canonical known evaluation time.
    # Input: bond=['A','A'], ts=['2026-01-02 10:00','2026-01-02 10:00'], no ID column.
    # Output: row_id=[0,1], cusip=['A','A'], time=[10:00-05:00,10:00-05:00] on 2026-01-02.
    # Explanation: Distinct transactions may share bond/time; row identity is never inferred from that pair.
    # Trick: Generated IDs follow source row order before filtering; a supplied ID must be unique and nonmissing.
    frame['row_id'] = original[columns.id] if columns.id else np.arange(len(frame), dtype='int64')
    frame['cusip'] = _identifier(original[columns.bond])
    frame['time'] = _times(original[columns.time], config.timezone, 'transaction time')
    id_blank = frame['row_id'].astype('string').str.strip().eq('').fillna(False)
    invalid_keys = frame['row_id'].isna() | id_blank | frame['cusip'].isna() | frame['time'].isna()
    if invalid_keys.any() or frame['row_id'].duplicated().any():
        raise ValueError(f'Transactions need valid unique row IDs and nonmissing bond/time; invalid_key_rows={int(invalid_keys.sum())}, duplicate_id_rows={int(frame.row_id.duplicated().sum())}.')
    # CORE LOGIC: STEP 2 — Normalize labels and optional numerical inputs without changing the population.
    # Input: target=[1.2,'bad',0], quantity=[1000,None,0], target_scale=100, quantity_scale=1000.
    # Output: target=[120,NaN,0], QUANTITY=[1000000,NaN,0]; all three trades remain.
    # Explanation: A missing label prevents supervised fitting for that trade, not its use in time-causal metadata.
    # Trick: The same target scale applies to anchor/cpp so differences stay in one declared value unit.
    frame['target'] = _number(original[columns.target], config.target_scale)
    numeric = {'QUANTITY': columns.quantity, 'PREV_QUANTITY': columns.prev_quantity,
               'anchor': columns.anchor, 'cpp': columns.cpp, 'MATURITY_YEARS': columns.maturity_years}
    for destination, source in numeric.items():
        if source is not None:
            scale = config.quantity_scale if 'QUANTITY' in destination else (config.target_scale if destination in {'anchor', 'cpp'} else 1.)
            frame[destination] = _number(original[source], scale)
    # CORE LOGIC: STEP 3 — Add declared grouping metadata and maturity without inventing missing groups.
    # Input: issuer=[' ACME ',None], sector=['TMT',''], no maturity mapping or split mapping.
    # Output: ISSUER=['ACME',<NA>], SECTOR=['TMT',<NA>]; existing source columns remain present.
    # Explanation: Missing metadata stays missing for explicit diagnosis and later grouping decisions.
    # Trick: Source-column collisions were rejected before assignment, protecting arbitrary baseline features.
    for destination, source in [('ISSUER', columns.issuer), ('SECTOR', columns.sector)]:
        if source is not None:
            frame[destination] = _identifier(original[source])
    if columns.maturity_date is not None:
        frame['MATURITY_DATE'], years = _maturity_years(original, columns, frame['time'], config.timezone)
        if columns.maturity_years is None:
            frame['MATURITY_YEARS'] = years
    if columns.split is not None:
        frame['split'] = _split(original[columns.split])
    # REPORTING LOGIC: Counts describe supplied records; no nonfinite-label row was silently removed.
    diagnostics = dict(input_rows=len(frame), output_rows=len(frame), excluded_rows=0, invalid_key_rows=0,
                       invalid_target_rows=int(frame.target.isna().sum()), generated_row_ids=columns.id is None,
                       bonds=int(frame.cusip.nunique()), timezone=config.timezone, value_kind=config.value_kind,
                       target_scale=config.target_scale, error_scale=config.error_scale, unit=config.unit)
    diagnostics['unknown_quantity_rows'] = int((frame.QUANTITY.isna() | frame.QUANTITY.le(0)).sum()) if columns.quantity else len(frame)
    diagnostics['maturity_source'] = 'supplied_years' if columns.maturity_years else ('calendar_days/365.25' if columns.maturity_date else 'unavailable')
    frame.attrs['normalization'] = diagnostics.copy()
    return frame, diagnostics


def _quote_rows(frame, columns, config):
    # CORE LOGIC: STEP 1 — Identify original exact duplicates before assigning generated source IDs.
    # Input: original records [(A,D,10:00,100),(A,D,10:00,100),(A,D,10:01,101)], no source ID.
    # Output: raw_source_id=[0,1,2], source_duplicate=[False,True,False].
    # Explanation: A unique generated ID cannot make a repeated original observation look economically new.
    # Trick: keep='first' preserves a deterministic representative; duplicate status precedes normalization.
    duplicates = frame.duplicated(keep='first')
    source_ids = frame[columns.source_id] if columns.source_id else pd.Series(np.arange(len(frame)), index=frame.index)
    common = pd.DataFrame({'firm': _identifier(frame[columns.dealer]), 'cusip': _identifier(frame[columns.bond]),
                           'quote_timestamp_ET': _times(frame[columns.known_time], config.timezone, 'quote known_time'),
                           'raw_source_id': source_ids, 'source_duplicate': duplicates})
    if columns.original_timestamp is not None:
        common['original_timestamp'] = frame[columns.original_timestamp]
    # CORE LOGIC: STEP 2 — Expand each configured wide side without dropping an incomplete value or unknown size.
    # Input: bid=[0,-2], ask=[1,NaN], bid_size=[5,None], ask_size=[8,0], quote_scale=1, quote_quantity_scale=1.
    # Output: bid (value,size)=[(0,5),(-2,NaN)]; ask [(1,8),(NaN,0)], four rows total.
    # Explanation: Bid and ask get their own quantities; nonfinite values remain incomplete events.
    # Trick: Side-specific size overrides common quantity even when that side's size is missing; no guessed fallback.
    pieces = []
    for side, value, size in [('bid', columns.bid, columns.bid_size), ('ask', columns.ask, columns.ask_size)]:
        if value is not None:
            piece = common.copy()
            piece['side'], piece['spread'] = side, _number(frame[value], config.quote_scale)
            quantity = size if size is not None else columns.quantity
            piece['quantity'], piece['quantity_kind'], piece['quantity_raw'] = _quantity(frame, quantity, config.quote_quantity_scale)
            pieces.append(piece)
    if pieces:
        return pd.concat(pieces, ignore_index=True)
    # CORE LOGIC: STEP 3 — Normalize the explicit side of long-form records, retaining invalid sides for diagnosis.
    # Input: side=[' B ','offer','unknown'], value=[0,-2,3], quantity=[5,None,0].
    # Output: side=['bid','ask',NaN], spread=[0,-2,3], quantity=[5,NaN,0].
    # Explanation: Unknown directions cannot become bids or asks by guessing; the caller counts/excludes those rows.
    # Trick: Buy/sell transaction language is intentionally unsupported because its perspective can be ambiguous.
    common['side'] = _identifier(frame[columns.side]).str.lower().map({'b': 'bid', 'bid': 'bid', 'a': 'ask', 'ask': 'ask', 'offer': 'ask'})
    common['spread'] = _number(frame[columns.value], config.quote_scale)
    common['quantity'], common['quantity_kind'], common['quantity_raw'] = _quantity(frame, columns.quantity, config.quote_quantity_scale)
    return common


def normalize_quotes(data, config, transactions):
    """Return bid/ask events for transaction-universe bonds; preserve invalid values as incomplete events."""
    # VALIDATION LOGIC: Transactions must already provide the explicit traded-bond universe.
    if not isinstance(config, PipelineConfig):
        raise TypeError('config must be a PipelineConfig.')
    if not isinstance(transactions, pd.DataFrame) or 'cusip' not in transactions:
        raise ValueError('Pass normalized transactions containing cusip; do not infer the quote universe.')
    columns = config.quotes
    identifiers = [name for name in [columns.bond, columns.dealer, columns.source_id] if name]
    frame = read_frame(data, identifiers).reset_index(drop=True)
    _required(frame, columns)
    mappings = {'firm': columns.dealer, 'cusip': columns.bond, 'quote_timestamp_ET': columns.known_time,
                'side': columns.side, 'spread': columns.value, 'quantity': columns.quantity,
                'raw_source_id': columns.source_id, 'source_duplicate': None, 'quantity_kind': None, 'quantity_raw': None}
    _reserved(frame, mappings)
    rows = _quote_rows(frame, columns, config)
    # CORE LOGIC: STEP 1 — Restrict to traded bonds and valid event keys using the recorded known time.
    # Input: transaction cusip=['A']; quotes cusip=['A','B','A'], firm=['D','D',None], side=['bid','bid','ask'].
    # Output: only the first quote remains; outside_universe_rows=1, invalid_key_rows=1, excluded_rows=2.
    # Explanation: Quotes for B are outside the research universe; the last A row has no dealer identity.
    # Trick: Diagnostic reasons may overlap; excluded_rows alone is the exclusive count. Values/sizes do not filter.
    universe = set(transactions.cusip.dropna())
    outside = rows.cusip.notna() & ~rows.cusip.isin(universe)
    invalid = rows[['firm', 'cusip', 'side', 'quote_timestamp_ET']].isna().any(axis=1)
    kept = rows.loc[~outside & ~invalid].copy().reset_index(drop=True)
    # REPORTING LOGIC: Separate original source counts from expanded side-row diagnostics.
    diagnostics = dict(input_rows=len(frame), expanded_rows=len(rows), output_rows=len(kept),
                       excluded_rows=len(rows)-len(kept), invalid_key_rows=int(invalid.sum()),
                       outside_universe_rows=int(outside.sum()), source_duplicate_rows=int(frame.duplicated().sum()),
                       invalid_value_rows=int(kept.spread.isna().sum()),
                       unknown_quantity_rows=int((kept.quantity.isna() | kept.quantity.le(0)).sum()),
                       zero_value_rows=int(kept.spread.eq(0).sum()), negative_value_rows=int(kept.spread.lt(0).sum()),
                       timezone=config.timezone, value_kind=config.value_kind, quote_scale=config.quote_scale,
                       quote_quantity_scale=config.quote_quantity_scale, time_source=columns.known_time,
                       original_timestamp_used_for_state=False, same_day_state=True,
                       limitation='Source diagnostic reasons may overlap; quotes are not converted between price and spread.')
    kept.attrs['normalization'] = diagnostics.copy()
    return kept, diagnostics
