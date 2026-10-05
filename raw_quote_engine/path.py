"""Eight causal path features from an existing event table; no raw loading or fitting."""
# SETUP LOGIC: Shared numeric dependencies and fixed feature contract.
import numpy as np
import pandas as pd
from .state import aware_time

# CONFIGURATION LOGIC: Window and freshness default to 30 minutes; support counts are diagnostics, not model columns.
PATH_VERSION = 'guarded-path-v1'
PATH_SUFFIXES = ('last_move_signed_median', 'flat_refresh_share',
                 'widened_30m_share', 'narrowed_30m_share')
PATH_FEATURES = [f'bcq_path_{side}_{suffix}' for side in ('bid', 'ask') for suffix in PATH_SUFFIXES]
SUPPORT_COLUMNS = [f'bcq_path_{side}_comparable_n' for side in ('bid', 'ask')]
PATH_DEFINITIONS = {
    'last_move_signed_median': {'unit': 'quote value unit', 'definition': 'Median last nonzero guarded quote-value move across currently comparable dealers; move may predate the configured lookback window.'},
    'flat_refresh_share': {'unit': 'fraction', 'definition': 'Share of currently comparable dealers whose latest quote/quantity pairs equal their preceding message.'},
    'widened_30m_share': {'unit': 'fraction', 'definition': 'Share of comparable dealers whose last nonzero spread move is positive within the configured left-open lookback window.'},
    'narrowed_30m_share': {'unit': 'fraction', 'definition': 'Share of comparable dealers whose last nonzero spread move is negative within the configured left-open lookback window.'}}
_MINUTE = 60 * 10**9
_NAT = np.iinfo(np.int64).min


def _series(group):
    # CORE LOGIC: STEP 1 — Order each dealer and compare consecutive complete states.
    # Input: A at 10:00/10:05/10:10 has center=[100,103,103], quantity=[2,2,2], count=[1,1,1].
    # Output: times=[10:00ns,10:05ns,10:10ns], delta=[NaN,3,0], comparable=[False,True,True].
    # Explanation: The first state has no predecessor; differences thereafter are 103-100 and 103-103.
    # Trick: Sorting is per bond/side/day/dealer; no cross-day predecessor enters this function.
    g = group.sort_values('quote_timestamp_ET', kind='stable')
    times = g.quote_timestamp_ET.array.as_unit('ns').asi8
    center = pd.to_numeric(g.center, errors='coerce').to_numpy(dtype=float)
    complete = g.complete.fillna(False).to_numpy(dtype=bool) & np.isfinite(center)
    delta = np.r_[np.nan, np.diff(center)]
    comparable = np.r_[False, complete[1:] & complete[:-1] & (np.diff(times) <= 60 * _MINUTE)]
    # VALIDATION LOGIC: Reject ambiguous duplicate event identities instead of choosing an arbitrary row.
    if len(times) > 1 and (np.diff(times) <= 0).any():
        raise ValueError('Path features require unique firm/bond/side/known-time events')
    # CORE LOGIC: STEP 2 — Break history whenever quote conditions change.
    # Input: centers=[100,103,104,104], quantity=[2,2,3,3], counts=[1,1,1,1].
    # Output: guarded delta=[NaN,3,NaN,0], segments=[1,1,2,2].
    # Explanation: 103 to 104 changes quantity, so its apparent +1 cannot count as a comparable move.
    # Trick: Tuple quantity labels compare identity, including zero/unknown; they are not trade-size units.
    quantities = g.quantity_set.map(lambda value: tuple(value))
    same = quantities.eq(quantities.shift()).to_numpy()
    same &= g.candidate_count.eq(g.candidate_count.shift()).to_numpy()
    comparable &= same
    delta[~comparable] = np.nan
    segments = np.cumsum(~comparable)
    # CORE LOGIC: STEP 3 — Carry only a previously observed true move within its condition segment.
    # Input: guarded delta=[NaN,3,-2,0,NaN,0] at minutes=[0,5,10,15,20,25]; pair_refresh=[False,False,False,True,False,True].
    # Output: last_move=[NaN,3,-2,-2,NaN,NaN], last_time=[NaT,5,10,10,NaT,NaT], refresh=[False,False,False,True,False,True].
    # Explanation: The 15-minute refresh retains -2; the break at 20 clears the prior segment.
    # Trick: Prefix ffill is causal; a zero center delta is a refresh only when the full quote/quantity pairs match.
    changed = np.isfinite(delta) & (delta != 0)
    moved = pd.Series(np.where(changed, delta, np.nan)).groupby(segments).ffill().to_numpy()
    moved_time = pd.Series(g.quote_timestamp_ET.array).where(changed).groupby(segments).ffill()
    last_ns = moved_time.array.as_unit('ns').asi8
    refresh = g.pair_refresh.fillna(False).to_numpy(dtype=bool) & comparable
    return times, complete, delta, moved, last_ns, refresh


def _at(states, query_ns, age_min=30, lookback_min=30, allow_exact=True):
    # CORE LOGIC: STEP 1 — Select the latest message before checking its completeness and freshness.
    # Input: A has complete100 at10:00 and incomplete at10:10; query=10:15.
    # Output: index=1, active=False; no fallback to the older complete quote.
    # Explanation: With default settings right-search includes equal known-time messages; age <=30min alone cannot validate incomplete state.
    # Trick: A query before the first event uses safe index zero but is masked out by index>=0.
    latest, moves, recent, active_columns = [], [], [], []
    for times, complete, delta, moved, last_ns, refresh in states:
        index = np.searchsorted(times, query_ns, side='right' if allow_exact else 'left') - 1
        safe = np.maximum(index, 0)
        active = (index >= 0) & complete[safe] & (query_ns - times[safe] <= age_min * _MINUTE)
        active &= np.isfinite(delta[safe])
        latest.append(active & refresh[safe])
        moves.append(np.where(active, moved[safe], np.nan))
        recent.append(active & (last_ns[safe] != _NAT) & (last_ns[safe] > query_ns - lookback_min * _MINUTE))
        active_columns.append(active)
    # CORE LOGIC: STEP 2 — Allocate one row per target, including targets with no dealer support.
    # Input: query_ns=[10:15ns,10:20ns], states=[];
    # Output: values=[[NaN,NaN,NaN,NaN],[NaN,NaN,NaN,NaN]], comparable_n=[0,0].
    # Explanation: Empty support is known zero; unknown direction must not become a zero movement.
    # Trick: All output arrays use query order, not event order.
    output = np.full((len(query_ns), 4), np.nan)
    count = np.zeros(len(query_ns), dtype=np.int64)
    if not states:
        return output, count
    refresh_mask, true_moves = np.column_stack(latest), np.column_stack(moves)
    known, is_recent = np.column_stack(active_columns), np.column_stack(recent)
    count = known.sum(axis=1)
    # CORE LOGIC: STEP 3 — Give each comparable dealer one vote, with median amplitude robust to outliers.
    # Input: refresh_mask=[True,False,False], last_true=[3,-2,1], recent=[True,True,False] for one query.
    # Output: [last_move_median=1, flat_share=1/3, widened_share=1/3, narrowed_share=1/3], count=3.
    # Explanation: Median(3,-2,1)=1; the flat dealer's last true move was recent +3 and still votes widened.
    # Trick: Shares use all currently comparable dealers; last-move median uses only known true moves.
    has_move = np.isfinite(true_moves).any(axis=1)
    output[has_move, 0] = np.nanmedian(true_moves[has_move], axis=1)
    indicators = [refresh_mask, is_recent & (true_moves > 0), is_recent & (true_moves < 0)]
    for column, mask in enumerate(indicators, 1):
        np.divide(mask.sum(axis=1), count, out=output[:, column], where=count > 0)
    return output, count


def build_path_features(events, frame, positions=None, progress=None, age_min=30, lookback_min=30, allow_exact=True):
    """Build only requested rows; default is the original Train/Validation cohort."""
    # VALIDATION LOGIC: Reuse a checked event table; never silently reconstruct raw quotes.
    if not np.isfinite(age_min) or age_min <= 0 or not np.isfinite(lookback_min) or lookback_min < 0:
        raise ValueError('age_min must be positive finite and lookback_min nonnegative finite.')
    event_frame = events.get('events') if isinstance(events, dict) else events
    required = {'cusip', 'side', 'firm', 'quote_timestamp_ET', 'center', 'complete',
                'quantity_set', 'candidate_count', 'pair_refresh'}
    if not isinstance(event_frame, pd.DataFrame) or not required.issubset(event_frame):
        raise ValueError('Restore the existing complete event cache before building path features')
    if not {'row_id', 'cusip', 'time', 'split'}.issubset(frame) or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError('Path features need unique original row_id and original target time/split')
    # CORE LOGIC: STEP 1 — Preserve requested identities without relabeling Test or dropping no-quote targets.
    # Input: row_id=[10,11,12], split=['Train','Validation','Test'], positions=None.
    # Output: output row_id=[10,11], original splits preserved; explicit positions=[2] returns only row12/Test.
    # Explanation: The default mask includes Train and Validation; final evaluation supplies its exact missing positions.
    # Trick: positions are positional indices, not DataFrame index labels, and must be unique valid integers.
    if positions is None:
        positions = np.flatnonzero(frame['split'].isin(['Train', 'Validation']).to_numpy())
    positions = np.asarray(positions)
    # VALIDATION LOGIC: Reject repeated/out-of-bounds indices rather than duplicating evaluation targets.
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError('Path query positions must be a one-dimensional integer array')
    if len(set(positions)) != len(positions) or (positions < 0).any() or (positions >= len(frame)).any():
        raise ValueError('Path query positions must be unique and in frame bounds')
    # CORE LOGIC: STEP 2 — Prepare fixed-width outputs and query keys once.
    # Input: row11/X at2026-03-02 10:15 ET; no quote has yet been selected.
    # Output: one row with eight NaN path values, bid/ask comparable_n=0; query day=2026-03-02 ET.
    # Explanation: Every requested target is retained before quote matching; only supported cells will be overwritten.
    # Trick: Convert datetime resolution explicitly to ns; NaT never becomes a valid epoch query.
    output = frame.iloc[positions][['row_id', 'cusip', 'time', 'split']].reset_index(drop=True).copy()
    output[PATH_FEATURES] = np.nan
    output[SUPPORT_COLUMNS] = 0
    times = aware_time(output.time)
    queries = pd.DataFrame({'cusip': output.cusip, '_day': times.dt.normalize(), '_ns': times.array.as_unit('ns').asi8})
    valid = queries._ns.ne(_NAT)
    # PROGRESS LOGIC: Show the indexing phase before scanning cached event timestamps and building groups.
    if progress is not None:
        progress('path_index', None, None, f'Indexing existing events for {len(output):,} Path queries; no raw aggregation')
    # CORE LOGIC: STEP 3 — Restrict event indexing to requested bonds and local dates, preserving prefix history.
    # Input: queries X on03-02; events X03-01, X03-02 09:00/11:00, Y03-02.
    # Output: event index retains X03-02 09:00/11:00 only; per-query asof later excludes future11:00.
    # Explanation: Whole requested days retain earlier true moves needed through refreshes, without carrying across days.
    # Trick: Do not truncate at the lookback boundary: the last true move can predate it while current messages remain fresh.
    event_times = aware_time(event_frame.quote_timestamp_ET)
    days = event_times.dt.normalize()
    selected = event_frame.cusip.isin(queries.loc[valid, 'cusip']) & days.isin(queries.loc[valid, '_day'])
    narrow = event_frame.loc[selected, sorted(required)].copy()
    narrow['quote_timestamp_ET'] = event_times.loc[selected]
    narrow['_day'] = days.loc[selected]
    event_index = narrow.groupby(['cusip', 'side', '_day'], sort=False, observed=True).indices
    query_index = queries.loc[valid].groupby(['cusip', '_day'], sort=False, observed=True).groups
    # PROGRESS LOGIC: The counter reflects bond/day groups, including cheap no-quote paths.
    total = len(query_index)
    if progress is not None:
        progress('path', 0, total, 'Reusing events for guarded last-move/refresh features')
    # CORE LOGIC: STEP 4 — Reuse each dealer history across bounded query blocks and map back by position.
    # Input: X/day queries at positions[3,1], bid dealers A/B; no ask events.
    # Output: bid path values write to rows3/1 at their own times; ask remains NaN with support0.
    # Explanation: Each event group is converted once, then blocks of <=512 queries reuse those arrays.
    # Trick: Bounded query-by-dealer matrices avoid materializing the full target-by-dealer product.
    for completed, ((bond, day), indices) in enumerate(query_index.items(), 1):
        rows = np.asarray(indices, dtype=int)
        for side in ('bid', 'ask'):
            event_rows = event_index.get((bond, side, day), [])
            if not len(event_rows):
                continue
            group = narrow.iloc[event_rows]
            states = [_series(g) for _, g in group.groupby('firm', sort=False, observed=True)]
            columns = [f'bcq_path_{side}_{suffix}' for suffix in PATH_SUFFIXES]
            # CORE LOGIC: STEP 5 — Preserve query row identity while filling only matching side columns.
            # Input: block rows=[3,1], computed bid values=[[2,0,1,0],[-1,0,0,1]], support=[2,1].
            # Output: output row3 bid=[2,0,1,0]/n2; row1 bid=[-1,0,0,1]/n1; other rows unchanged.
            # Explanation: loc assigns a positional result array to explicit original output row numbers.
            # Trick: NumPy assignment avoids unintended pandas label alignment between a block and the full output.
            for start in range(0, len(rows), 512):
                block = rows[start:start + 512]
                values, counts = _at(states, queries.loc[block, '_ns'].to_numpy(dtype=np.int64), age_min, lookback_min, allow_exact)
                output.loc[block, columns] = values
                output.loc[block, f'bcq_path_{side}_comparable_n'] = counts
        # PROGRESS LOGIC: Report bounded milestones and the actual final group.
        if progress is not None and (completed % max(1, total // 100) == 0 or completed == total):
            progress('path', completed, total, f'Path bond/days {completed:,}/{total:,}')
    # REPORTING LOGIC: Carry definitions and source cache identity with the standalone sidecar.
    metadata = {'version': PATH_VERSION, 'features': PATH_FEATURES, 'support_columns': SUPPORT_COLUMNS,
                'feature_dictionary': {f'bcq_path_{side}_{suffix}': dict(spec, side=side)
                                       for side in ('bid', 'ask') for suffix, spec in PATH_DEFINITIONS.items()},
                'rows': len(output), 'window_min': lookback_min, 'fresh_age_min': age_min, 'history_gap_min': 60,
                'source_event_cache_key': events.get('cache_key') if isinstance(events, dict) else None,
                'allow_exact': bool(allow_exact), 'timezone': str(times.dt.tz),
                'definition': 'Same local day; latest complete fresh state with the declared exact-time policy; condition breaks reset center-move history. Refresh requires identical quote/quantity pairs.',
                'limitations': 'Raw quantity equality is not trade-size matching; last true move may predate the configured lookback; no target values used. Legacy _30m names use window_min.'}
    output.attrs.update(metadata)
    return output, metadata
