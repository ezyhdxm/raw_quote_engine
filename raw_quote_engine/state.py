# SETUP LOGIC: Module documentation and imports; importing neither loads raw data nor trains models.
"""Candidate-set state engine adapted from the audited BondCliQ research.

The internal spread field and _bps suffixes denote the configured numeric quote
unit. Only spread input is supported; users prepare benchmark-consistent spreads upstream.
"""
import numpy as np
import pandas as pd
from time import perf_counter

# SETUP LOGIC: Fixed field names, thresholds and feature definitions; no observations are filtered or estimated here.
KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
SERIES = KEYS[:3]
HISTORY_GAP_MIN = 60
LOOKBACK_MIN = 30
DEFAULT_AGE_MIN = 30
MIN_PEERS = 3
CLIP_FLOOR_BPS = 10.0
MAD_MULTIPLIER = 4.0


def aware_time(series):
    # TIME CONVERSION LOGIC: Ingestion already localized known times; preserve that timezone and calendar.
    values = pd.to_datetime(series, errors="raise", format="mixed")
    if values.dt.tz is None:
        raise ValueError("Normalize timestamps before building quote features.")
    return values


def event_history(raw, progress=None, keep_raw=True):
    """Distinct sets and prefix-only changes; optional progress(stage, done, total, detail)."""
    # PROGRESS LOGIC: Record stage clocks and optional progress without changing event values.
    started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Normalizing {len(raw):,} quote rows')
    # CORE LOGIC: STEP 1 — Preserve full-row duplicate semantics and identify valid event keys
    # Input: rows=[('A','X','bid',10:00,100,2),('A','X','bid',10:00,100,2),(' ','X','bid',10:00,100,2)].
    # Output: repeat=[False,True,False]; valid=[True,True,False].
    # Explanation: The second complete source row duplicates the first; the third has a blank dealer key.
    # Trick: source_duplicate is computed on original full rows before source IDs; strip validates keys without rewriting them.
    repeats = raw["source_duplicate"] if "source_duplicate" in raw else raw.duplicated()
    quantity_metadata = [name for name in ['quantity_kind', 'quantity_raw'] if name in raw]
    q = raw.copy() if keep_raw else raw[KEYS + ['spread', 'quantity'] + quantity_metadata].copy()
    valid = q[KEYS].notna().all(axis=1)
    for key in SERIES:
        valid &= q[key].astype("string").str.strip().ne("").fillna(False)
    q["repeat"] = repeats
    # CORE LOGIC: STEP 2 — Convert spread values and classify quantity
    # Input: spread=[0,-5,inf,'bad']; quantity=[0,2,None,-1].
    # Output: s=[0,-5,NaN,NaN]; qkind=['Zero','Positive','Missing','Other'].
    # Explanation: Finite zero and negative spreads remain; quantity categories distinguish original nulls from invalid values.
    # Trick: Only nonfinite spread becomes NaN; unknown quantity does not delete a record.
    q["s"] = pd.to_numeric(q["spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    q["q"] = pd.to_numeric(q["quantity"], errors="coerce")
    if 'quantity_kind' in q:
        q['qkind'] = q['quantity_kind']
    else:
        masks = [q["quantity"].isna(), q["q"].eq(0), np.isfinite(q["q"]) & q["q"].gt(0)]
        q["qkind"] = np.select([mask.fillna(False).to_numpy(dtype=bool) for mask in masks],
                               ["Missing", "Zero", "Positive"], default="Other")
    # CORE LOGIC: STEP 3 — Build comparable quantity labels
    # Input: quantity is an object column [2,0,None,-1].
    # Output: qtag=['q=2.0','Zero','Missing','Other:-1'].
    # Explanation: Positive 2 uses its float representation; zero/null retain their categories and Other retains the original string.
    # Trick: Stable float repr makes equal positive quantities comparable; quantity_raw retains distinct invalid source tokens after numeric coercion.
    q["qtag"] = q["qkind"].astype(str)
    positive = q["qkind"].eq("Positive")
    q.loc[positive, "qtag"] = q.loc[positive, "q"].map(lambda v: "q=" + repr(float(v)))
    other = q["qkind"].eq("Other")
    raw_quantity = q['quantity_raw'] if 'quantity_raw' in q else q['quantity']
    q.loc[other, "qtag"] = "Other:" + raw_quantity.loc[other].astype(str)
    # CORE LOGIC: STEP 4 — Retain incomplete candidates and all keyed rows
    # Input: s=[100,NaN]; qtag=['q=2.0','Missing']; valid=[True,True].
    # Output: pair=[(100,'q=2.0'),('Nonfinite','Missing')]; usable has both rows.
    # Explanation: Mark the second spread bad and preserve it through a sentinel when constructing its candidate pair.
    # Trick: Dropping bad rows first would falsely turn an incomplete event into a complete one.
    q["bad"] = q["s"].isna()
    q["pair"] = list(zip(q["s"].fillna("Nonfinite"), q["qtag"]))
    usable = q.loc[valid]
    # PROGRESS LOGIC: Record normalization time and report how many keyed rows will be aggregated.
    normalized = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Aggregating {len(usable):,} keyed quote rows')
    # CORE LOGIC: STEP 5 — Aggregate distinct candidates by dealer/bond/side/known time
    # Input: A/X/bid/10:00 rows=(100,2),(110,2),(100,2),(NaN,None).
    # Output: rows=4; repeats=1; bad=1; spread_set=(100,110); quantity_set=('Missing','q=2.0'); candidate_count=2.
    # Explanation: One event keeps two distinct finite spreads and both quantity labels; the duplicate contributes only to repeat count.
    # Trick: Distinct sets prevent repeated rows gaining price weight; stable sorting keeps each series chronological.
    g = usable.groupby(KEYS, observed=True, sort=False).agg(
        rows=("s", "size"), repeats=("repeat", "sum"), bad=("bad", "sum"),
        spread_set=("s", lambda v: tuple(sorted(v.dropna().unique()))),
        quantity_set=("qtag", lambda v: tuple(sorted(v.unique()))),
        pair_set=("pair", lambda v: frozenset(v)),
    ).reset_index().sort_values(SERIES + [KEYS[-1]], kind="stable").reset_index(drop=True)
    g["candidate_count"] = g["spread_set"].map(len)
    # PROGRESS LOGIC: Record aggregation time and announce the history stage.
    aggregated = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Building history for {len(g):,} events')
    # CORE LOGIC: STEP 6 — Describe candidate bounds, median and completeness
    # Input: event1 spread_set=(100,110),bad=0; event2 spread_set=(),bad=1.
    # Output: event1 lo=100,hi=110,center=105,gap=10,center_nearest_gap=5,complete=True; event2 center=NaN,complete=False.
    # Explanation: The two-candidate median is 105, with both candidates 5 bps away; an empty event has no price.
    # Trick: A median need not be an original quote; any bad candidate makes its event incomplete.
    for col, fn in [("lo", min), ("hi", max), ("center", np.median)]:
        g[col] = g["spread_set"].map(lambda v: float(fn(v)) if v else np.nan)
    g["gap"] = g["hi"] - g["lo"]
    g["center_nearest_gap"] = [min(abs(s - c) for s in ss) if ss else np.nan
                               for ss, c in zip(g["spread_set"], g["center"])]
    g["complete"] = g["bad"].eq(0) & g["candidate_count"].gt(0)
    g["day"] = g[KEYS[-1]].dt.normalize()
    # CORE LOGIC: STEP 7 — Align the previous event within each dealer/bond/side/local day
    # Input: A/X/bid: day1 10:00 complete100,10:10 complete105; day2 10:00 complete106.
    # Output: day1 10:10 interval_min=10,continuous=True; day2 first interval_min=NaN,history_break=True.
    # Explanation: Grouped shift supplies the earlier same-day event; the next day starts a new group without a predecessor.
    # Trick: shift aligns original indices; first events, incomplete endpoints and gaps above 60 minutes break history.
    group = g.groupby(SERIES + ["day"], sort=False, observed=True)
    prev = group[["spread_set", "quantity_set", "pair_set", "complete", "candidate_count", "center", KEYS[-1]]].shift()
    g["interval_min"] = (g[KEYS[-1]] - prev[KEYS[-1]]).dt.total_seconds() / 60
    continuous = g["interval_min"].le(HISTORY_GAP_MIN) & g["complete"] & prev["complete"].eq(True)
    g["history_break"] = ~continuous
    # CORE LOGIC: STEP 8 — Separate refreshes, spread moves and changed quote conditions
    # Input: previous spread=(100,),quantity=('q=2.0',),count=1; current spread=(105,),quantity=('q=3.0',),count=1; continuous=True.
    # Output: spread_changed=True; pair_refresh=False; condition_changed=True; center_delta=5; guarded_delta=NaN.
    # Explanation: Center rises by 5, but changing quantity makes the move ineligible for the condition-stable statistic.
    # Trick: guarded requires equal quantity sets and candidate counts; signed raw movement is still retained.
    g["spread_changed"] = continuous & g["spread_set"].ne(prev["spread_set"])
    g["pair_refresh"] = continuous & g["pair_set"].eq(prev["pair_set"])
    g["condition_changed"] = continuous & (g["quantity_set"].ne(prev["quantity_set"]) | g["candidate_count"].ne(prev["candidate_count"]))
    g["center_delta"] = (g["center"] - prev["center"]).where(continuous)
    g["guarded_delta"] = g["center_delta"].where(~g["condition_changed"])
    # CORE LOGIC: STEP 9 — Measure stable-condition boundary changes and observed ABA
    # Input: Same-day complete quotes 100→105→100, quantity=2 throughout, adjacent gaps=10min.
    # Output: Third event lo_delta=-5,hi_delta=-5,observed_aba=True.
    # Explanation: Both boundaries fall by 5; the third pair matches two events back, differs from its predecessor, and both intervals are continuous.
    # Trick: Bounds are order statistics, not tracked quote identities; ABA requires two continuous intervals.
    for bound in ["lo", "hi"]:
        g[f"{bound}_delta"] = group[bound].diff().where(continuous & ~g["condition_changed"])
    two_back = group["pair_set"].shift(2)
    g["observed_aba"] = continuous & group["history_break"].shift().eq(False) & g["pair_set"].eq(two_back) & g["pair_set"].ne(prev["pair_set"])
    # CORE LOGIC: STEP 10 — Compute prefix-only change age within history segments
    # Input: Same segment: 10:00 first100,10:05 change105,10:10 refresh105.
    # Output: history_start=[10:00,10:00,10:00]; last_change=[NaT,10:05,10:05]; change_age_min=[NaN,0,5].
    # Explanation: The first observed change at 10:05 is carried forward to 10:10, never backward to 10:00.
    # Trick: cumsum isolates breaks; ffill uses only observed past changes; first-event age remains unknown.
    segment = g["history_break"].cumsum()
    g["history_start"] = g.groupby(segment)[KEYS[-1]].transform("first")
    g["last_change"] = g[KEYS[-1]].where(g["spread_changed"]).groupby(segment).ffill()
    g["change_age_min"] = (g[KEYS[-1]] - g["last_change"]).dt.total_seconds() / 60
    g["change_age_unknown"] = g["last_change"].isna()
    # CORE LOGIC: STEP 11 — Extract nanosecond times and contiguous segment boundaries once
    # Input: history_break=[True,False,True]; spread_changed=[False,True,False].
    # Output: starts=[0,2]; stops=[2,3]; changed=[0,1,0]; initial changes_30m=[0,0,0].
    # Explanation: Break positions delimit slices [0:2] and [2:3]; preallocate counts in the existing event order.
    # Trick: as_unit('ns') prevents timestamp rounding from admitting a future event; contiguous slices avoid repeated indexing.
    event_times = g[KEYS[-1]].array.as_unit("ns").asi8
    changed = g["spread_changed"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(g["history_break"].to_numpy())
    stops = np.r_[starts[1:], len(g)]
    changes_30m = np.zeros(len(g), dtype=np.int64)
    # PROGRESS LOGIC: Announce the history-segment count without changing features.
    if progress is not None:
        progress('events', 0, len(starts), 'Counting changes within history segments')
    # CORE LOGIC: STEP 12 — Count changes in the left-open 30-minute window
    # Input: One segment at 10:00,10:05,10:35 with changed=[0,1,1].
    # Output: changes_30m at 10:35=1; moving the last timestamp to 10:34:59.999999999 gives 2.
    # Explanation: At 10:35 the left boundary equals 10:05 and is excluded; a one-nanosecond-earlier query includes it.
    # Trick: searchsorted(right) enforces (t-30min,t]; prefix differences include the current change; singletons stay zero.
    for completed, (start, stop) in enumerate(zip(starts, stops), 1):
        # A singleton starts with history_break=True, hence spread_changed=False.
        if stop-start > 1:
            times = event_times[start:stop]
            counts = np.r_[0, changed[start:stop].cumsum()]
            left = np.searchsorted(times, times - LOOKBACK_MIN * 60 * 10**9, side="right")
            changes_30m[start:stop] = counts[1:] - counts[left]
        # PROGRESS LOGIC: Throttle progress notifications by completed segment count.
        if progress is not None and (completed % max(1, (len(starts) + 99) // 100) == 0 or completed == len(starts)):
            progress('events', completed, len(starts), f'History segments {completed:,}/{len(starts):,}')
    # CORE LOGIC: STEP 13 — Write counts once and return event metadata
    # Input: Three sorted events, changes_30m=[0,1,1]; valid=[True,True,True,False]; keep_raw=False; clocks=0,1,2,3 seconds.
    # Output: events.changes_30m=[0,1,1]; unkeyed=1; timings={normalize_s:1,aggregate_s:1,history_s:1}; no raw key.
    # Explanation: Assign by the same sorted positions and count one invalid source key; each adjacent clock difference is one second.
    # Trick: Timings are observation metadata, never model features; keep_raw controls only the returned raw copy.
    g["changes_30m"] = changes_30m
    result = {"events": g, "unkeyed": int((~valid).sum()),
              "timings": {'normalize_s': normalized-started,
                          'aggregate_s': aggregated-normalized,
                          'history_s': perf_counter()-aggregated}}
    if keep_raw:
        result['raw'] = q
    return result


def prepare_quote_events(quotes, progress=None):
    """Reusable narrow event table. Rebuild explicitly after changing source quotes.

    It retains incomplete latest messages; an incomplete message is observed state,
    not an absent quote. No hidden cache can silently reuse a changed DataFrame.
    """
    # CACHEING LOGIC: Reuse event construction without a wide raw copy; retain incomplete latest messages.
    return event_history(quotes, progress=progress, keep_raw=False)




def pair_snapshots(events, times, age_min=30, sync_min=1, allow_exact=True, value_kind="spread"):
    """One bond; latest bid/ask per dealer, same configured local day. Cartesian ranges stay set-valued."""
    # VALIDATION LOGIC: Require positive age and nonnegative synchronization tolerance.
    if value_kind != 'spread':
        raise ValueError('Only spread quotes are supported; convert source prices upstream.')
    if age_min <= 0 or sync_min < 0: raise ValueError('age_min must be positive and sync_min nonnegative')
    # CORE LOGIC: STEP 1 — Sort and deduplicate query times for one bond
    # Input: events contains only X; times=[10:05,10:00,10:05]; age_min=30,sync_min=1.
    # Output: times=[10:00,10:05].
    # Explanation: One calculation at each distinct time suffices; caller-owned trade identities remain separate.
    # Trick: Deduplication reduces computation only; results can still join to every original target.
    times = pd.DatetimeIndex(times).sort_values().unique()
    # VALIDATION LOGIC: Reject multiple CUSIPs in a single-bond as-of calculation.
    if events['cusip'].nunique() > 1:
        raise ValueError('pair_snapshots expects one bond')
    # SETUP LOGIC: Collect dealer-side as-of rows using a fixed state-field schema.
    rows = []
    fields = ['firm','side',KEYS[-1],'day','complete','center','lo','hi','candidate_count','pair_set']
    # CORE LOGIC: STEP 2 — Select the latest same-day message per dealer and side
    # Input: A bid09:59 complete100,10:04 incomplete; query10:05.
    # Output: Latest bid timestamp=10:04,complete=False,age=1; no fallback to09:59.
    # Explanation: Backward as-of chooses the latest known message before checking completeness.
    # Trick: allow_exact=False excludes equal known time; incomplete latest messages block older valid state.
    for (firm, side), h in events.groupby(['firm','side'], observed=True):
        if side not in ['bid','ask']: continue
        joined = pd.merge_asof(pd.DataFrame({'time':times}), h[fields].sort_values(KEYS[-1]),
                               left_on='time', right_on=KEYS[-1], direction='backward', allow_exact_matches=allow_exact)
        joined = joined.loc[joined[KEYS[-1]].notna() & joined.time.dt.normalize().eq(joined.day)].copy()
        joined['age'] = (joined.time - joined[KEYS[-1]]).dt.total_seconds()/60
        rows.append(joined)
    # CORE LOGIC: STEP 3 — Outer-join bid and ask by query time and dealer
    # Input: slots=(10:05,A,bid,100),(10:05,B,bid,105),(10:05,B,ask,102).
    # Output: pairs=(10:05,A,center_bid100,center_askNaN),(10:05,B,center_bid105,center_ask102).
    # Explanation: A keeps its single-sided state while B's two sides share one row.
    # Trick: outer retains missing sides; different dealers cannot form a pair.
    fields += ['time','age']
    slots = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=fields)
    b = slots.loc[slots.side.eq('bid')].drop(columns='side')
    a = slots.loc[slots.side.eq('ask')].drop(columns='side')
    p = b.merge(a,on=['time','firm'],how='outer',suffixes=('_bid','_ask'))
    # CORE LOGIC: STEP 4 — Derive both-side presence, completeness, age and synchronization
    # Input: query10:05; A bid10:04 complete100,ask10:03 complete102; B has no ask.
    # Output: A both=True,complete=True,max_age=2,time_gap=1,same_time=False; B both=False.
    # Explanation: Ages are 1 and 2 minutes, and side timestamps differ by 1 minute.
    # Trick: Time gap is absolute minutes; zero/negative spreads do not invalidate completeness.
    p['both'] = p[f'{KEYS[-1]}_bid'].notna() & p[f'{KEYS[-1]}_ask'].notna()
    p['complete'] = p.complete_bid.eq(True) & p.complete_ask.eq(True)
    p['max_age'] = p[['age_bid','age_ask']].max(axis=1).where(p.both)
    p['time_gap'] = (pd.to_datetime(p[f'{KEYS[-1]}_bid']) - pd.to_datetime(p[f'{KEYS[-1]}_ask'])).abs().dt.total_seconds()/60
    p['same_time'] = p.both & p['time_gap'].eq(0)
    # CORE LOGIC: STEP 5 — Initialize unassessed pair values and support counts
    # Input: B has bid100 and no ask.
    # Output: B gap_low/gap_high/gap_center/mid=NaN; positive_matches=0; size_state='Unassessed'.
    # Explanation: Missing support has no measurable price difference; only the absence of matching support has count zero.
    # Trick: Never fill unknown gaps with zero.
    for name in ['gap_low','gap_high','gap_center','mid','mid_range','matched_gap_low','matched_gap_high','matched_gap','matched_mid']:
        p[name] = np.nan
    p['positive_matches'] = 0
    p['size_state'] = 'Unassessed'
    # CORE LOGIC: STEP 6 — Compute signed bid-minus-ask Cartesian bounds
    # Input: bid candidates100/110,ask105/115; centers105/110.
    # Output: gap_low=-15,gap_high=5,gap_center=-5,mid=107.5,mid_range=10.
    # Explanation: Bounds use100-115 and110-105; center gap is105-110 and midpoint is(105+110)/2.
    # Trick: Signed bounds retain crossing; candidate ranges preserve every distinct finite quote.
    p['gap_low'] = (p.lo_bid-p.hi_ask).where(p.complete)
    p['gap_high'] = (p.hi_bid-p.lo_ask).where(p.complete)
    p['gap_center'] = (p.center_bid-p.center_ask).where(p.complete)
    p['mid'] = ((p.center_bid+p.center_ask)/2).where(p.complete)
    p['mid_range'] = ((p.hi_bid-p.lo_bid+p.hi_ask-p.lo_ask)/2).where(p.complete)
    # Repeated snapshots reuse the same candidate-set match, instead of expanding pairs.
    # CACHEING LOGIC: Memoize matching by immutable bid/ask candidate sets instead of repeatedly expanding identical states.
    cache = {}
    matched_columns = ['size_state','positive_matches','matched_gap_low','matched_gap_high','matched_gap','matched_mid']
    matches = []
    # CORE LOGIC: STEP 7 — Group each complete pair's candidates by raw quantity tag
    # Input: bid={(100,'q=2.0'),(110,'q=2.0'),(120,'Zero')}; ask={(105,'q=2.0')}.
    # Output: bid groups q=2.0→[100,110],Zero→[120]; ask q=2.0→[105], ignoring list order.
    # Explanation: Collect spreads under their original tags before checking positive-quantity intersections.
    # Trick: Zero/unknown candidates remain in the original sets; frozenset iteration does not guarantee list order.
    for r in p.loc[p.complete].itertuples():
        key = (r.pair_set_bid, r.pair_set_ask)
        if key not in cache:
            bid, ask = {}, {}
            for pairs, dest in [(r.pair_set_bid,bid),(r.pair_set_ask,ask)]:
                for spread, tag in pairs: dest.setdefault(tag,[]).append(spread)
            # CORE LOGIC: STEP 8 — Identify shared positive quantity and matching state
            # Input: pair1 bid tags=['q=2.0','Zero'],ask=['q=2.0']; pair2 bid=['q=2.0'],ask=['q=3.0'].
            # Output: pair1 shared=['q=2.0'],state='Shared positive'; pair2 state='Positive unmatched'.
            # Explanation: The first intersection has a positive tag; the second has none although both sides are positive.
            # Trick: Unknown/mixed is distinct from unmatched positive quantity; absence of a match does not mean all sizes are missing.
            shared = [tag for tag in set(bid)&set(ask) if tag.startswith('q=')]
            only_positive = all(tag.startswith('q=') for tag in list(bid)+list(ask))
            state = 'Shared positive' if shared else ('Positive unmatched' if only_positive else 'Unknown / mixed')
            values = [state,0,np.nan,np.nan,np.nan,np.nan]
            # CORE LOGIC: STEP 9 — Calculate bounds and median gaps within shared quantity
            # Input: bid q=2 candidates100/110; ask q=2 candidate105; only q=2 is shared.
            # Output: positive_matches=1; matched_gap_low=-5,matched_gap_high=5,matched_gap=0,matched_mid=105.
            # Explanation: The bid median105 equals ask105; extreme differences are100-105 and110-105.
            # Trick: Preserve multiprice within each quantity, then equally average shared-quantity summaries; quantity is not a weight.
            if shared:
                values = [state,len(shared), min(min(bid[q])-max(ask[q]) for q in shared),
                    max(max(bid[q])-min(ask[q]) for q in shared),
                    np.mean([np.median(bid[q])-np.median(ask[q]) for q in shared]),
                    np.mean([(np.median(bid[q])+np.median(ask[q]))/2 for q in shared])]
            # CACHEING LOGIC: Store/reuse the candidate-set match so repeated grid states do not recompute it.
            cache[key] = values
        matches.append(cache[key])
    # CORE LOGIC: STEP 10 — Write matched statistics back using original pair indices
    # Input: complete pair indices=[1,3]; matched gaps=[0,2].
    # Output: p.loc[[1,3],'matched_gap']=[0,2]; incomplete rows remain NaN.
    # Explanation: The temporary result frame explicitly carries indices1 and3 before assignment.
    # Trick: Matching indices prevent pandas from aligning results to unrelated rows.
    if matches:
        p.loc[p.complete,matched_columns] = pd.DataFrame(matches,index=p.index[p.complete],columns=matched_columns)
    # CORE LOGIC: STEP 11 — Classify crossing for original and matched candidate sets
    # Input: gap ranges=[0,5],[-5,-1],[-5,5]; fourth pair incomplete.
    # Output: cross=['None','All','Some','Unassessed']; absent positive matches also remain Unassessed.
    # Explanation: Nonnegative low means no crossing; negative high means all combinations cross; straddling zero means some cross.
    # Trick: low>=0 includes a locked zero gap; high<0 is strict; classification never deletes candidates.
    for suffix,lo,hi,eligible in [('', 'gap_low','gap_high',p.complete),
                                  ('_matched','matched_gap_low','matched_gap_high',p.positive_matches.gt(0))]:
        p['cross'+suffix] = 'Unassessed'
        p.loc[eligible,'cross'+suffix] = np.select([p.loc[eligible,lo].ge(0),p.loc[eligible,hi].lt(0)],['None','All'],default='Some')
    # CORE LOGIC: STEP 12 — Sort pair rows and derive policy eligibility
    # Input: A at10:05 is complete,max_age=2,time_gap=1,positive_matches=1; age_min=30,sync_min=1.
    # Output: A fresh_pair=True,size_time_pair=True; pairs sorted by time,firm.
    # Explanation: Age2<=30, synchronization1<=1 and one positive match satisfy both masks.
    # Trick: Policy masks select views without deleting original candidates or crossing evidence.
    p = p.sort_values(['time','firm'],kind='stable').reset_index(drop=True)
    return pair_policy_masks(p, age_min, sync_min)


def pair_policy_masks(pairs, age_min=30, sync_min=1):
    """Derive age/sync eligibility from cached all-dealer state, without as-of work."""
    # VALIDATION LOGIC: Require positive age and nonnegative synchronization tolerance for policy masks.
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    # CORE LOGIC: STEP 1 — Derive fresh and size-time masks from cached pairs
    # Input: A complete,max_age=30,time_gap=1,matches=1; B complete,max_age=31,time_gap=1,matches=1; age=30,sync=1.
    # Output: A fresh_pair=True,size_time_pair=True; B fresh_pair=False,size_time_pair=False.
    # Explanation: A satisfies the inclusive age/sync limits; B fails freshness before quantity eligibility matters.
    # Trick: copy protects the cache; changing cutoffs requires no new as-of work.
    p = pairs.copy()
    p['fresh_pair'] = p.complete & p.max_age.le(age_min)
    p['size_time_pair'] = p.fresh_pair & p.time_gap.le(sync_min) & p.positive_matches.gt(0)
    return p


def pair_policy_comparison(pairs):
    """A→B isolates slot selection; B→C isolates candidate matching on identical slots.

    Slots are all-dealer bond/day grid observations, not independent trades. B is
    eligible for size/time matching but uses its original candidate sets. C uses
    shared positive raw-quantity candidates on exactly those same B slots.
    """
    # CORE LOGIC: STEP 1 — Separate original-slot selection from same-slot matching
    # Input: dealerA fresh=True,size_time=False,gap=-10,mid95,cross='All'; dealerB fresh=True,size_time=True,gap4,mid103,cross='Some',matched_gap2,matched_mid99,cross_matched='None'.
    # Output: policyA: n=2,dealers=2,gap=-3,mid99,cross(None/Some/All)=(0,.5,.5); B: n=1,gap4,mid103,cross=(0,1,0); C: n=1,gap2,mid99,cross=(1,0,0).
    # Explanation: A averages both slots; B selects dealerB; C changes candidates on that identical B slot.
    # Trick: A→B gap+7 is selection; B→C gap-2 is matching; grid slots are not independent trades.
    rows = []
    for label, mask, cross, gap, mid in [
        ('A fresh / original', pairs.fresh_pair, 'cross', 'gap_center', 'mid'),
        ('B match slots / original', pairs.size_time_pair, 'cross', 'gap_center', 'mid'),
        ('C same slots / matched', pairs.size_time_pair, 'cross_matched', 'matched_gap', 'matched_mid')]:
        z = pairs.loc[mask]
        rows.append(dict(policy=label, n_slots=len(z), n_dealers=z.firm.nunique(),
            gap_mean_bps=z[gap].mean(), mid_mean_bps=z[mid].mean(),
            **{state.lower()+'_fraction': z[cross].eq(state).mean() for state in ['None', 'Some', 'All']}))
    # CORE LOGIC: STEP 2 — Return one indexed row per policy
    # Input: records in policy,n_slots,n_dealers,gap,mid,None/Some/All order: (A,2,2,-3,99,0,.5,.5),(B,1,1,4,103,0,1,0),(C,1,1,2,99,1,0,0).
    # Output: index=[A,B,C]; numeric rows=[[2,2,-3,99,0,.5,.5],[1,1,4,103,0,1,0],[1,1,2,99,1,0,0]].
    # Explanation: DataFrame construction preserves record order and moves policy labels into the index.
    # Trick: B and C use exactly the same size_time_pair mask, so their counts match.
    return pd.DataFrame(rows).set_index('policy')




def pair_features(pairs, times):
    """Dealer-equal observed gaps; unknown quantities never become matched-size pairs."""
    # SETUP LOGIC: Declare distinct query times and the pair-feature schema.
    index = pd.DatetimeIndex(times).sort_values().unique()
    cols = ['n_pair','n_size_time_pair','pair_gap','pair_gap_low','pair_gap_high','pair_mid',
            'pair_mid_range','pair_cross_some','pair_cross_all','pair_time_gap','pair_unknown_size',
            'size_time_gap','size_time_mid','size_time_cross_some','size_time_cross_all']
    # CORE LOGIC: STEP 1 — Initialize unknown pair values with zero support counts
    # Input: times=[10:00,10:05]; no pair rows.
    # Output: Both rows n_pair=n_size_time_pair=0; every gap,mid,range,crossing fraction,time gap and unknown-size value is NaN.
    # Explanation: Preallocate the full output schema, then assign zero only to support counts.
    # Trick: No support and an observed zero gap have different meanings.
    f = pd.DataFrame(np.nan,index=index,columns=cols)
    f[['n_pair','n_size_time_pair']] = 0
    # CORE LOGIC: STEP 2 — Select policy slots and encode crossing
    # Input: At10:05 freshA cross='None',freshB cross='All'; a third slot is not fresh.
    # Output: fresh q containsA/B; cross_some=[0,0],cross_all=[0,1].
    # Explanation: The policy mask drops only the third diagnostic slot; categorical comparisons encode each retained state.
    # Trick: Each dealer slot gets one vote; matched policy uses cross_matched on its own declared mask.
    for eligible,prefix,cross_field,gap_field,mid_field in [
        ('fresh_pair','pair','cross','gap_center','mid'),
        ('size_time_pair','size_time','cross_matched','matched_gap','matched_mid')]:
        q = pairs.loc[pairs[eligible]].copy()
        if q.empty: continue
        q['cross_some'] = q[cross_field].eq('Some').astype(float)
        q['cross_all'] = q[cross_field].eq('All').astype(float)
        # CORE LOGIC: STEP 3 — Count each time's slots and map output fields
        # Input: q has dealersA/B at10:05 andA at10:06.
        # Output: n_pair(10:05)=2,n_pair(10:06)=1; pair_gap maps to gap_center.
        # Explanation: Group size counts rows per time and label-based assignment restores their matching output rows.
        # Trick: counts.index and f.loc align explicitly instead of relying on group iteration order.
        g = q.groupby('time',sort=False)
        counts = g.size()
        count_column = 'n_pair' if prefix=='pair' else 'n_size_time_pair'
        f.loc[counts.index,count_column] = counts
        mapping = {prefix+'_gap':gap_field,prefix+'_mid':mid_field,
                   prefix+'_cross_some':'cross_some',prefix+'_cross_all':'cross_all'}
        # CORE LOGIC: STEP 4 — Describe absent positive matches, range and synchronization
        # Input: At10:05 A positive_matches=1,time_gap=0; B positive_matches=0,time_gap=2.
        # Output: pair_unknown_size=.5,pair_time_gap=1; low/high/mid_range retain their source mappings.
        # Explanation: No-match indicators [0,1] average to.5; the median time gap of[0,2] is1.
        # Trick: No shared positive quantity includes unmatched positive tags; it does not mean all quantities were missing.
        if prefix=='pair':
            q['unknown_size'] = q.positive_matches.eq(0).astype(float)
            g = q.groupby('time',sort=False)
            mapping.update(pair_gap_low='gap_low',pair_gap_high='gap_high',
                           pair_mid_range='mid_range',pair_unknown_size='unknown_size')
            f.loc[counts.index,'pair_time_gap'] = g.time_gap.median()
        # CORE LOGIC: STEP 5 — Aggregate each dealer with equal weight
        # Input: At10:05 A gap=-4,B gap=2; cross_all=[0,1].
        # Output: pair_gap=-1,pair_cross_all=.5; f.index.name='time'.
        # Explanation: Average(-4+2)/2=-1 and(0+1)/2=.5, then align the summaries to the time index.
        # Trick: Raw repetition and quantity do not affect dealer weights; unsupported times remain NaN.
        for output,source in mapping.items(): f.loc[counts.index,output] = g[source].mean()
    f.index.name='time'
    return f


def empty_quote_features(times):
    """The same zero-count/NaN schema as real as-of calculations, with no dealer work."""
    # SETUP LOGIC: Match the real side calculation's output columns and query index.
    index = pd.DatetimeIndex(times).sort_values().unique()
    sides = ['center_equal','n_dealers','n_fresh_dealers','n_incomplete','center_decay',
             'center_max_age','dispersion_bps','mean_candidate_gap','multi_fraction',
             'zero_quantity_fraction','unknown_quantity_fraction','median_message_age_min',
             'median_change_age_min','unknown_change_age_fraction','center_lower','center_upper',
             'max_decay_weight_share','decay_effective_dealers','center_candidate_clip',
             'center_dealer_downweight','n_peer_supported','n_clipped_dealers','n_changed_centers']
    # CORE LOGIC: STEP 1 — Keep zero integer support and unknown values for both sides
    # Input: times=['2026-03-02 10:00-05:00','2026-03-02 10:05-05:00']; no dealer state.
    # Output: For both times/sides, n_dealers,n_fresh_dealers,n_incomplete,n_peer_supported,n_clipped_dealers,n_changed_centers are int0; every other side value is NaN.
    # Explanation: Build a NaN frame per side and replace only the six count columns with integer zeros.
    # Trick: Match side_features_fast dtypes; latest-incomplete messages still require the real calculation path.
    counts = ['n_dealers','n_fresh_dealers','n_incomplete','n_peer_supported','n_clipped_dealers','n_changed_centers']
    pieces = []
    for side in ['bid', 'ask']:
        f = pd.DataFrame(np.nan, index=index, columns=sides)
        # Match side_features_fast's integer count dtypes as well as its values.
        for count in counts:
            f[count] = np.zeros(len(index), dtype=int)
        pieces.append(f.add_prefix('bcq_'+side+'_'))
    # CORE LOGIC: STEP 2 — Combine the empty side and pair schemas
    # Input: times=[10:00,10:05]; both side frames have zero support and NaN centers; empty_pairs has0 rows.
    # Output: Both times bcq_n_pair=0,bcq_n_size_time_pair=0,bcq_has_quote=0.0; every pair/size-time gap,mid,range or fraction is NaN.
    # Explanation: Empty pair_features uses the same times; concatenate all pieces along their common index.
    # Trick: Identical prefixes and schema make later label-based writes safe.
    empty_pairs = pd.DataFrame(columns=['fresh_pair', 'size_time_pair'])
    pieces.append(pair_features(empty_pairs, index).add_prefix('bcq_'))
    result = pd.concat(pieces, axis=1)
    result['bcq_has_quote'] = 0.0
    result.index.name = 'time'
    return result


def build_quote_features(quotes, queries, age_min=30, sync_min=1, allow_exact=True, progress=None, event_cache=None, value_kind="spread", clip_floor=10.0):
    """Preserve every query. Optional prepare_quote_events result avoids repeated event work.

    Timings and skipped no-state query counts are attached to result.attrs. Empty
    bond/day and before-first queries skip both side matrices and pair snapshots;
    latest incomplete messages still use the full path to preserve n_incomplete.
    """
    # VALIDATION LOGIC: Reject duplicate target identities, missing query keys and invalid age/sync parameters.
    if value_kind != 'spread':
        raise ValueError('Only spread quotes are supported; convert source prices upstream.')
    if queries.row_id.duplicated().any() or queries[['row_id','cusip','time']].isna().any().any():
        raise ValueError('Queries require unique row_id and nonmissing cusip/time')
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    # SETUP LOGIC: Record clocks/cache reuse and group queries by CUSIP.
    started = perf_counter()
    cache_reused = event_cache is not None
    query_groups=queries.groupby('cusip',sort=False,observed=True)
    # PROGRESS LOGIC: Report original target count and distinct queried bonds.
    if progress is not None:
        progress('features', 0, query_groups.ngroups, f'{len(queries):,} trade queries across {query_groups.ngroups:,} bonds')
    # FAST PATH LOGIC: Return an empty result and timings immediately when no queries exist; do not prepare events.
    if queries.empty:
        result = queries.copy()
        result.attrs['quote_feature_timings'] = dict(event_prepare_s=0.0, asof_s=0.0,
            total_s=perf_counter()-started, no_state_unique_queries=0, event_cache_reused=cache_reused)
        return result
    # CORE LOGIC: STEP 1 — Prepare relevant-bond events only if no checked cache exists
    # Input: quotes containsX/Y; queries contains onlyX; event_cache=None.
    # Output: Only X quote rows reach prepare_quote_events; supplied event_cache instead supplies its existing events unchanged.
    # Explanation: The queried CUSIP set narrows uncached source processing; a cache avoids another raw scan.
    # Trick: The caller must bind cached events to the correct source identity.
    if event_cache is None:
        relevant = quotes.loc[quotes.cusip.isin(queries.cusip.unique())]
        event_cache = prepare_quote_events(relevant, progress=progress)
    all_events = event_cache['events']
    # SETUP LOGIC: Record event preparation time and initialize output/event groups by bond.
    prepared = perf_counter()
    output=[]
    grouped=all_events.groupby('cusip',observed=True)
    skipped = 0
    # PROGRESS LOGIC: Report feature progress by queried bond; the denominator is distinct query CUSIPs.
    for completed,(bond,q) in enumerate(query_groups, 1):
        if progress is not None:
            progress('features', completed-1, query_groups.ngroups, f'Bond {bond}: {len(q):,} trade queries')
        # CORE LOGIC: STEP 2 — Fast-path queries before the first same-day event
        # Input: X first quote isMar2 10:00; queries=Mar2 09:59,Mar2 10:00,Mar3 10:00; allow_exact=True.
        # Output: active=[False,True,False]; active_times=[Mar2 10:00]; skipped increases2; other query counts0 and centersNaN.
        # Explanation: Map each query day to its first event; only one query reaches a same-day observation.
        # Trick: Latest-incomplete state after the first event is still active, not a no-quote shortcut.
        events=grouped.get_group(bond) if bond in grouped.groups else all_events.iloc[:0]
        times=pd.DatetimeIndex(q.time).sort_values().unique()
        first = events.groupby('day', observed=True)[KEYS[-1]].min()
        first_at_query = pd.Series(times.normalize(), index=times).map(first)
        active = first_at_query.notna() & (first_at_query.le(times) if allow_exact else first_at_query.lt(times))
        active_times = times[active.to_numpy()]
        skipped += int((~active).sum())
        f = empty_quote_features(times)
        # CORE LOGIC: STEP 3 — Calculate both sides and pairs only at active distinct times
        # Input: A/X at10:00 bid100(q2),ask105(q2); queries=[09:59,10:00],age30,sync1,exact=True.
        # Output: At10:00 centers=100/105,side dealer counts=1/1,n_pair=n_size_time_pair=1,pair_gap=size_time_gap=-5;09:59 retains0 counts and NaN values.
        # Explanation: Both complete sides share positive quantity and timestamp, giving signed gap100-105=-5.
        # Trick: Blocks contain at most 512 distinct times, bounding dealer matrices; duplicate-time trades are restored later.
        for start in range(0, len(active_times), 512):
            block = active_times[start:start + 512]
            pieces=[]
            for side in ['bid','ask']:
                side_f=side_features_fast(events.loc[events.side.eq(side)],block,age_min,allow_exact,clip_floor)
                pieces.append(side_f.add_prefix(f'bcq_{side}_'))
            p=pair_snapshots(events,block,age_min,sync_min,allow_exact,value_kind)
            pieces.append(pair_features(p,block).add_prefix('bcq_'))
            # CORE LOGIC: STEP 4 — Write active rows into the complete query-time schema
            # Input: f@09:59/10:00 has has_quote0,bid_n0,bid_centerNaN; active_f@10:00 has bid_n1,ask_n0,bid_center100.
            # Output: f@10:00 has has_quote1,bid_n1,bid_center100;09:59 remains has_quote0,bid_n0,bid_centerNaN.
            # Explanation: One available side makes total dealer support positive; only the matching active time is overwritten.
            # Trick: f.loc[block] aligns DatetimeIndex and columns with this block's active_f; other blocks remain untouched.
            active_f=pd.concat(pieces,axis=1)
            active_f['bcq_has_quote']=(active_f.bcq_bid_n_dealers.add(active_f.bcq_ask_n_dealers).gt(0)).astype(float)
            f.loc[block] = active_f
        # CORE LOGIC: STEP 5 — Join unique-time features back to every original trade
        # Input: q=[{row_id:7,cusip:'X',time:10:00},{row_id:8,cusip:'X',time:10:00}]; f@10:00 has_quote=1.
        # Output: joined has both row_id7 and8 at10:00, each with cusipX and has_quote1.
        # Explanation: Each trade independently receives the single feature row for its time.
        # Trick: many_to_one rejects duplicate feature times while retaining repeated trade timestamps.
        joined=q[['row_id','cusip','time']].merge(f,left_on='time',right_index=True,how='left',validate='many_to_one')
        output.append(joined)
        # PROGRESS LOGIC: Report the number of original trade queries completed for this bond.
        if progress is not None:
            progress('features', completed, query_groups.ngroups, f'Finished bond {bond}: {len(q):,} trade queries')
    # CORE LOGIC: STEP 6 — Restore original row order and record stage timings
    # Input: queries row_id=[8,7]; output rows=(7,X,10:00,has_quote1),(8,Y,10:00,has_quote0); clocks0/1/2s,skipped1,cache_reused=True.
    # Output: result order=(8,Y,10:00,0),(7,X,10:00,1); timings event_prepare_s1,asof_s1,total_s2,no_state_unique_queries1,event_cache_reusedTrue.
    # Explanation: Set row_id as the index and reindex by the caller's original order; adjacent clock differences supply timing metadata.
    # Trick: Group concatenation order cannot replace identity-based reindexing.
    result = pd.concat(output,ignore_index=True).set_index('row_id').reindex(queries.row_id).reset_index() if output else queries.copy()
    result.attrs['quote_feature_timings'] = dict(event_prepare_s=prepared-started,
        asof_s=perf_counter()-prepared, total_s=perf_counter()-started,
        no_state_unique_queries=skipped, event_cache_reused=cache_reused,
        **event_cache.get('timings', {}))
    return result


def side_features_fast(events, times, age_min=30, allow_exact=True, clip_floor=10.0):
    """Vectorized query-by-dealer summaries; same definitions as Step 3, without momentum."""
    # VALIDATION LOGIC: Require a positive age parameter for both half-life and cutoff calculations.
    if age_min <= 0: raise ValueError('age_min must be positive')
    # CORE LOGIC: STEP 1 — Allocate query-by-dealer matrices and a nanosecond query axis
    # Input: times=[10:05,10:00,10:05]; events have dealersA/B.
    # Output: unique times=[10:00,10:05],n=2,d=2; numeric matrices shape(2,2) containNaN; present/valid containFalse.
    # Explanation: Distinct sorted times form rows and dealers form columns before any observations are assigned.
    # Trick: as_unit('ns') preserves exact ordering; deduplication affects computation only.
    times=pd.DatetimeIndex(times).sort_values().unique()
    groups=list(events.groupby('firm',observed=True)); n,d=len(times),len(groups)
    names=['center','lo','hi','gap','count','age','change_age','zero','unknown','mid_low','mid_high']
    a={k:np.full((n,d),np.nan) for k in names}
    present=np.zeros((n,d),bool); valid=np.zeros((n,d),bool)
    query_ns=times.as_unit('ns').asi8; days=times.normalize().asi8
    # CORE LOGIC: STEP 2 — Choose each dealer's latest event before checking completeness
    # Input: A10:00 complete100,10:04 incomplete; query10:05.
    # Output: idx selects10:04; present=True,valid=False; no fallback to10:00.
    # Explanation: The same-day latest message is present even when its completeness flag is false.
    # Trick: searchsorted right allows exact time, left excludes it; clamped -1 indices are later masked out.
    for j,(_,h) in enumerate(groups):
        h=h.sort_values(KEYS[-1]); stamp=h[KEYS[-1]].array.as_unit('ns').asi8
        idx=np.searchsorted(stamp,query_ns,side='right' if allow_exact else 'left')-1
        z=h.iloc[np.maximum(idx,0)]
        present[:,j]=(idx>=0)&(z.day.array.as_unit(times.unit).asi8==days)
        valid[:,j]=present[:,j]&z.complete.to_numpy(dtype=bool)
        # CORE LOGIC: STEP 3 — Copy candidate statistics and message/change ages
        # Input: A10:00 complete,center105,lo100,hi110,last_change09:55; query10:05.
        # Output: center105,lo100,hi110,message age5,change age10 minutes.
        # Explanation: Values come from the selected event; ages subtract each relevant timestamp from the query.
        # Trick: NaT last_change remains unknown; as-of selection uses integer nanoseconds before age conversion.
        for key,col in [('center','center'),('lo','lo'),('hi','hi'),('gap','gap'),('count','candidate_count')]:
            a[key][:,j]=z[col].to_numpy()
        a['age'][:,j]=(query_ns-stamp[np.maximum(idx,0)])/60e9
        last=z.last_change.array.as_unit('ns').asi8
        a['change_age'][:,j]=np.where(z.last_change.notna(),(query_ns-last.astype(float))/60e9,np.nan)
        # CORE LOGIC: STEP 4 — Preserve quantity status and the two middle candidates
        # Input: A spread_set=(100,110),quantity_set=('Zero','Missing').
        # Output: zero=True,unknown=True,mid_low=100,mid_high=110.
        # Explanation: Detect both quantity flags and retain lower/upper median positions from the sorted candidate set.
        # Trick: Clipping the middle candidates separately reconstructs the median of clipped candidates exactly.
        a['zero'][:,j]=z.quantity_set.map(lambda x:'Zero' in x).to_numpy()
        a['unknown'][:,j]=z.quantity_set.map(lambda x:any(t=='Missing' or t.startswith('Other:') for t in x)).to_numpy()
        for key,offset in [('mid_low',-1),('mid_high',0)]:
            values=h.spread_set.map(lambda x:x[(len(x)+offset)//2] if x else np.nan).to_numpy()
            a[key][:,j]=values[np.maximum(idx,0)]
    # CORE LOGIC: STEP 5 — Mask invalid values and derive fresh support
    # Input: A latest incomplete; B complete with age30; age_min=30.
    # Output: A numeric values becomeNaN; B valid=True,fresh=True; count=1.
    # Explanation: Only B contributes complete support, while A remains eligible for the incomplete-message count.
    # Trick: present differs from valid; inclusive <= retains the freshness boundary.
    for k in a: a[k][~valid]=np.nan
    count=valid.sum(axis=1); fresh=valid&(a['age']<=age_min)
    # AGGREGATION LOGIC: Helper for equal-dealer means; unsupported rows remain NaN.
    def mean(x,mask=valid):
        # CORE LOGIC: STEP 1 — Average only supported dealers; leave empty rows unknown
        # Input: x=[[100,NaN],[NaN,NaN]],mask=[[True,False],[False,False]].
        # Output: [100,NaN].
        # Explanation: First row divides100 by1; the second has no support and retains its prefilled NaN.
        # Trick: nansum gives0 for empty rows, but where=mask.sum>0 prevents an invented zero mean.
        return np.divide(np.nansum(np.where(mask,x,np.nan),axis=1),mask.sum(axis=1),out=np.full(n,np.nan),where=mask.sum(axis=1)>0)
    # AGGREGATION LOGIC: Helper for supported row medians; skip all-NaN rows.
    def median(x):
        # CORE LOGIC: STEP 1 — Calculate medians only where finite support exists
        # Input: x=[[100,110],[NaN,NaN]].
        # Output: [105,NaN].
        # Explanation: The first median is(100+110)/2; the all-missing row never enters nanmedian.
        # Trick: Checking finite support avoids all-NaN median warnings and preserves unknown values.
        result=np.full(n,np.nan); have=np.isfinite(x).any(axis=1)
        if have.any(): result[have]=np.nanmedian(x[have],axis=1)
        return result
    # CORE LOGIC: STEP 6 — Compute equal-dealer center and support counts
    # Input: A complete center100 age5; B complete center110 age35; C latest incomplete.
    # Output: center_equal=105,n_dealers=2,n_fresh_dealers=1,n_incomplete=1.
    # Explanation: Complete centers average to105; only A is fresh, while C is present but incomplete.
    # Trick: Quantity and repeated candidate rows never change dealer weights.
    f=pd.DataFrame(index=times)
    f['center_equal']=mean(a['center']); f['n_dealers']=count
    f['n_fresh_dealers']=fresh.sum(axis=1); f['n_incomplete']=(present&~valid).sum(axis=1)
    # CORE LOGIC: STEP 7 — Use numerically stable half-life weights
    # Input: A center100 age0; B center110 age30; age_min=30.
    # Output: weights=[1,.5]; center_decay≈103.333333; center_max_age=105.
    # Explanation: Weighted center is(100+55)/1.5; both ages meet the30-minute cutoff.
    # Trick: Subtracting each row's minimum age avoids common underflow without changing normalized weight ratios.
    min_age=np.min(np.where(valid,a['age'],np.inf),axis=1) if d else np.full(n,np.inf)
    weights=np.where(valid,np.exp2(-(a['age']-min_age[:,None])/age_min),0)
    weight_sum=weights.sum(axis=1)
    f['center_decay']=np.divide(np.nansum(a['center']*weights,axis=1),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['center_max_age']=mean(a['center'],fresh)
    # CORE LOGIC: STEP 8 — Summarize dispersion, candidate quantity and age reliability
    # Input: centers=[100,110],gaps=[0,10],counts=[1,2],ages=[5,15],zero=[T,F],unknown=[F,T],change_age=[NaN,10]; both valid.
    # Output: dispersion5,mean_candidate_gap5,multi/zero/unknown fractions=.5; median_message_age10,median_change_age10,unknown_change_age_fraction=.5.
    # Explanation: Population dispersion is sqrt((25+25)/2)=5; each flagged property affects one of two dealers.
    # Trick: Dispersion uses the equal-dealer center; quantity flags describe state and do not filter observations.
    f['dispersion_bps']=np.sqrt(mean((a['center']-f.center_equal.to_numpy()[:,None])**2))
    f['mean_candidate_gap']=mean(a['gap']); f['multi_fraction']=mean(a['count']>1)
    f['zero_quantity_fraction']=mean(a['zero']); f['unknown_quantity_fraction']=mean(a['unknown'])
    f['median_message_age_min']=median(a['age']); f['median_change_age_min']=median(a['change_age'])
    f['unknown_change_age_fraction']=mean(~np.isfinite(a['change_age']))
    # CORE LOGIC: STEP 9 — Report bounds and effective age-weight support
    # Input: A lo95,hi105,weight1; B lo105,hi115,weight.5.
    # Output: center_lower=100,center_upper=110,max_decay_weight_share=2/3,decay_effective_dealers=1.8.
    # Explanation: Bounds average by dealer; weight concentration is1/1.5 and effective support is1.5²/1.25.
    # Trick: Effective support is (sum w)²/sum(w²), distinct from the raw number of dealers.
    f['center_lower']=mean(a['lo']); f['center_upper']=mean(a['hi'])
    f['max_decay_weight_share']=np.divide(weights.max(axis=1) if d else np.zeros(n),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['decay_effective_dealers']=np.divide(weight_sum**2,(weights**2).sum(axis=1),out=np.full(n,np.nan),where=weight_sum>0)
    # CORE LOGIC: STEP 10 — Initialize leave-one-dealer-out rule state
    # Input: One query with validA/B/C/D centers=[100,101,102,200].
    # Output: clipped=[[100,101,102,200]],dw=[[1,1,1,1]]; supported/clipped_any/center_changed are allFalse.
    # Explanation: Start from original centers and equal valid weights; only supported peer comparisons may change them.
    # Trick: Unsupported dealers retain their original values; initialization is not validation of a rule.
    clipped=a['center'].copy(); dw=np.where(valid,1.0,0.0)
    supported=np.zeros((n,d),bool); clipped_any=supported.copy(); center_changed=supported.copy()
    # CORE LOGIC: STEP 11 — Build a peer reference from at least three other fresh dealers
    # Input: EvaluatedD center200; other fresh centers=[100,101,102].
    # Output: D ok=True,ref=101,MAD=1,radius=max(10,4*1.4826*1)=10.
    # Explanation: Three other dealers meet MIN_PEERS; their median is101 and median absolute deviation is1.
    # Trick: peer_mask[:,j]=False excludes self; fewer than3 peers preserves the original center.
    for j in range(d):
        peer_mask=fresh.copy(); peer_mask[:,j]=False
        ok=valid[:,j]&(peer_mask.sum(axis=1)>=MIN_PEERS)
        if not ok.any(): continue
        peers=np.where(peer_mask[ok],a['center'][ok],np.nan)
        ref=np.nanmedian(peers,axis=1)
        radius=np.maximum(clip_floor,MAD_MULTIPLIER*1.4826*np.nanmedian(abs(peers-ref[:,None]),axis=1))
        # CORE LOGIC: STEP 12 — Clip candidates and downweight original median residuals
        # Input: D candidates190/210,center200; peer ref101,radius10.
        # Output: clipped median111; residual99; dw=10/99;supported=True.
        # Explanation: Both middle candidates clip to the interval[91,111]; downweighting uses min(1,10/99).
        # Trick: Clip both median positions separately; residual weighting uses the original center and retains the dealer.
        clipped[ok,j]=(np.clip(a['mid_low'][ok,j],ref-radius,ref+radius)+np.clip(a['mid_high'][ok,j],ref-radius,ref+radius))/2
        residual=abs(a['center'][ok,j]-ref)
        dw[ok,j]=np.minimum(1,np.divide(radius,residual,out=np.ones(len(ref)),where=residual>0))
        supported[ok,j]=True
        clipped_any[ok,j]=(a['lo'][ok,j]<ref-radius)|(a['hi'][ok,j]>ref+radius)
        center_changed[ok,j]=~np.isclose(clipped[ok,j],a['center'][ok,j],rtol=0,atol=1e-9)
    # CORE LOGIC: STEP 13 — Combine rule centers and actual change/support counts
    # Input: centers=[100,101,102,200],clipped=[100,101,102,111],dw=[1,1,1,10/99]; all4 supported,onlyD changed.
    # Output: center_candidate_clip=103.5,center_dealer_downweight≈104.224756,n_peer_supported=4,n_clipped_dealers=1,n_changed_centers=1.
    # Explanation: Clipped mean is414/4; weighted mean is(303+200*10/99)/(3+10/99).
    # Trick: Unsupported originals remain included; change detection uses rtol0,atol1e-9 to suppress floating-point noise.
    f['center_candidate_clip']=mean(clipped)
    f['center_dealer_downweight']=np.divide(np.nansum(a['center']*dw,axis=1),dw.sum(axis=1),out=np.full(n,np.nan),where=dw.sum(axis=1)>0)
    f['n_peer_supported']=supported.sum(axis=1); f['n_clipped_dealers']=clipped_any.sum(axis=1); f['n_changed_centers']=center_changed.sum(axis=1)
    f.index.name='time'
    return f
