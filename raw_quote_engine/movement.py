# SETUP LOGIC: Module description and imports; importing does not load raw data or train models.
"""Causal matched-dealer and leave-one-bond-out issuer changes from known-time events.

The pipeline explicitly requests all rows, without reading targets. Internal _bps
suffixes are in configured quote units, including points for price input.
"""
# SETUP LOGIC: Read no data and fit no models on import.
from time import perf_counter

import numpy as np
import pandas as pd

from .state import aware_time


# SETUP LOGIC: Declare fixed fields, thresholds and feature names without filtering or estimating observations.
_SIDES = ("bid", "ask")
_MOVE_SUFFIXES = (
    "mean_bps", "median_bps", "guarded_mean_bps", "up_fraction",
    "down_fraction", "flat_fraction", "common_n", "current_n", "lookback_n",
    "retention", "condition_changed_fraction", "mean_support_age_min",
    "max_support_age_min", "guarded_common_n",
)
# SETUP LOGIC: Declare fixed issuer feature names without filtering or estimating observations.
_ISSUER_SUFFIXES = (
    "mean_bps", "median_bps", "up_fraction", "down_fraction", "flat_fraction",
    "other_bond_n", "common_dealer_n", "dispersion_bps",
    "mean_support_age_min", "max_support_age_min",
)
# SETUP LOGIC: Declare feature indices, count fields and fixed computational limits.
DIRECTION_FEATURES = [f"bcq_{s}_move_{v}" for s in _SIDES for v in _MOVE_SUFFIXES]
ISSUER_FEATURES = [f"bcq_{s}_issuer_move_{v}" for s in _SIDES for v in _ISSUER_SUFFIXES]
_MOVE_COL = {name: i for i, name in enumerate(_MOVE_SUFFIXES)}
_COUNT_SUFFIXES = {"common_n", "current_n", "lookback_n", "guarded_common_n",
                   "other_bond_n", "common_dealer_n"}
_MINUTE_NS = 60 * 10**9
_QUERY_BLOCK = 512
_NEVER = np.iinfo(np.int64).max


def _mean(values, n):
    # CORE LOGIC: STEP 1 — Divide by observed support counts, retaining NaN for empty rows.
    # Input: values=[[2,NaN],[NaN,NaN]],n=[1,0]
    # Output: [2,NaN]
    # Explanation: Only 2 is valid in the first row, giving sum 2 / support 1 = 2; the second row has no support and retains NaN, rather than treating nansum's 0 as an observed mean.
    # Trick: Although nansum returns 0 for the second row, where=n>0 prevents a zero mean without support.
    return np.divide(np.nansum(values, axis=1), n,
                     out=np.full(len(n), np.nan), where=n > 0)


def _median(values, n):
    # CORE LOGIC: STEP 1 — Compute medians only for supported rows.
    # Input: values=[[2,6],[NaN,NaN]],n=[2,0]
    # Output: [4,NaN]
    # Explanation: n>0 selects the first row; the middle two sorted values [2,6] average to 4, placed back in its row; the second row is never evaluated and remains NaN.
    # Trick: Select n>0 before nanmedian to avoid warnings on all-NaN rows and fabricated zeros.
    result = np.full(len(n), np.nan)
    ok = n > 0
    if ok.any():
        result[ok] = np.nanmedian(values[ok], axis=1)
    return result


def _maximum(values, n):
    # CORE LOGIC: STEP 1 — Compute maxima only for supported rows.
    # Input: values=[[2,6],[NaN,NaN]],n=[2,0]
    # Output: [6,NaN]
    # Explanation: The maximum of the first row's two supported values is 6; the unsupported second row skips max and retains NaN.
    # Trick: No support means unknown; an empty maximum age must not become 0.
    result = np.full(len(n), np.nan)
    ok = n > 0
    if ok.any():
        result[ok] = np.nanmax(values[ok], axis=1)
    return result


def _empty_move(n):
    # CORE LOGIC: STEP 1 — Construct a stable direction schema for unsupported rows.
    # Input: n=1, with no complete fresh dealer shared between endpoints.
    # Output: The sole row, ordered mean_bps,median_bps,guarded_mean_bps,up_fraction,down_fraction,flat_fraction,common_n,current_n,lookback_n,retention,condition_changed_fraction,mean_support_age_min,max_support_age_min,guarded_common_n, is [NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,NaN,NaN,0].
    # Explanation: Initialize all 14 outputs to NaN, then set only common/current/lookback/guarded counts to 0; absent quotes cannot be mistaken for unchanged direction.
    # Trick: A zero count means known absence of support; unknown movement, fractions and ages must not become 0.
    result = np.full((n, len(_MOVE_SUFFIXES)), np.nan)
    for col in _COUNT_SUFFIXES.intersection(_MOVE_COL):
        result[:, _MOVE_COL[col]] = 0
    return result


def _issuer_records(queries):
    """First known label and first observed conflict, never whole-sample backfill.

    Missing/unknown labels before the first known label do not establish a
    mapping. After that point, any different or unknown observation makes the
    mapping unreliable from its own timestamp onward. A later conflict cannot
    retroactively invalidate an earlier query.
    """
    # CORE LOGIC: STEP 1 — Normalize missing issuer labels while preserving query identities.
    # Input: queries has three X rows: 09:00 'Unknown',10:00 ' I ',11:00 'J'.
    # Output: labels=[NA,'I','J']; work retains each bond and known timestamp in integer nanoseconds.
    # Explanation: str.strip trims I; mask replaces Unknown with missing and retains J. Each row still matches its original bond/time; the later I cannot fill 09:00.
    # Trick: Only labels are stripped; CUSIP and query-time keys remain unchanged, and Unknown establishes no affiliation.
    labels = queries["ISSUER"].astype("string").str.strip()
    labels = labels.mask(labels.str.lower().isin(["", "unknown", "none", "nan", "<na>"]))
    work = pd.DataFrame({"bond": queries.cusip, "ns": queries._ns, "label": labels})
    records = {}
    # CORE LOGIC: STEP 2 — Establish mappings only from the earliest nonmissing, unambiguous timestamp.
    # Input: X has 09:00 Unknown,10:00 I,11:00 J; Y has both I and Unknown at 10:00.
    # Output: X first=10:00,label='I'; no record is created for Y.
    # Explanation: Group by bond. For X, the earliest known time is 10:00 and its sole label is I. For Y, I and missing coexist at the first known time, so affiliation is ambiguous and Y is skipped.
    # Trick: Unknown or conflicting labels at the same time are ambiguous; exclude the NaT sentinel before establishing mappings.
    for bond, g in work.loc[work.ns.ne(np.iinfo(np.int64).min)].groupby("bond", sort=False, observed=True):
        known = g.loc[g.label.notna()]
        if known.empty:
            continue
        first = int(known.ns.min())
        at_first = g.loc[g.ns.eq(first), "label"]
        if at_first.isna().any() or at_first.nunique() != 1:
            continue
        # CORE LOGIC: STEP 3 — Record the first observed conflict without rewriting past affiliation.
        # Input: X has 10:00 I,10:30 I,11:00 J; Z has 10:00 I with no later conflict.
        # Output: records['X']=('I',10:00ns,11:00ns); Z's conflict time is the maximum int64 value.
        # Explanation: X starts with I at 10:00; 10:30 agrees, while 11:00 J establishes expiry. Z has no later conflict, so the maximum integer represents no observed expiry.
        # Trick: Subsequent queries require first<=t<conflict; a conflict at 11:00 does not change the 10:30 result.
        label = str(at_first.iloc[0])
        conflict = g.loc[g.ns.ge(first) & (g.label.isna() | g.label.ne(label).fillna(True)), "ns"]
        records[bond] = (label, first, int(conflict.min()) if len(conflict) else _NEVER)
    return records


def _event_index(events, queries, needed_bonds, lookback_ns, age_ns):
    """One narrow index, restricted to queried local days and possible fresh times."""
    # CORE LOGIC: STEP 1 — Identify valid query nanoseconds and return early for no valid queries.
    # Input: queries contains only time=NaT; events contains X at 10:00.
    # Output: (pd.DataFrame() with 0 rows and 0 columns,{},set()); X's 10:00 event is not borrowed for the NaT query.
    # Explanation: Remove NaT's minimum-int64 sentinel; with no valid queries, return an empty index immediately. An X quote cannot create a feature for a nonexistent query timestamp.
    # Trick: NaT's minimum int64 is not time 0; with no actual query, there is no need to scan dealers.
    stamp = aware_time(events.quote_timestamp_ET).array.as_unit("ns").asi8
    day = aware_time(events.day).array.as_unit("ns").asi8
    valid_queries = queries.loc[queries._ns.ne(np.iinfo(np.int64).min)]
    if valid_queries.empty:
        return pd.DataFrame(), {}, set()
    # CORE LOGIC: STEP 2 — Retain only relevant dates, bonds, sides and potentially fresh past events.
    # Input: query X on March 2 at 10:30,needed_bonds={'X'},lookback=30 minutes,age=30 minutes; events X09:00/X10:00/X11:00/Y10:00.
    # Output: narrow contains only X10:00; index key (X,bid,March 2 midnight ns) points to row 0.
    # Explanation: 10:30 minus the 30-minute lookback and 30-minute age gives a 09:30 lower bound; X09:00 is too old, X11:00 is future and Y is not needed, leaving X10:00.
    # Trick: The global lower bound t_min-lookback-age removes messages that cannot be fresh; t_max excludes future events, while per-query dealer selection happens later.
    valid = (np.isin(day, valid_queries._day.unique()) &
             (stamp >= int(valid_queries._ns.min()) - lookback_ns - age_ns) &
             (stamp <= int(valid_queries._ns.max())) & events.cusip.isin(needed_bonds).to_numpy() &
             events.side.isin(_SIDES).to_numpy())
    cols = ["firm", "cusip", "side", "complete", "center", "quantity_set", "candidate_count"]
    narrow = events.loc[valid, cols].reset_index(drop=True)
    narrow["_ns"], narrow["_day"] = stamp[valid], day[valid]
    index = narrow.groupby(["cusip", "side", "_day"], observed=True, sort=False).indices
    return narrow, index, set(narrow._day.unique())


def _dealer_states(narrow, positions):
    # CORE LOGIC: STEP 1 — Sort each dealer's events within one bond/side/day slice.
    # Input: A messages arrive as 10:01 center102,10:00 center100; B has 10:00 center105.
    # Output: A's sorted times=[10:00ns,10:01ns]; two events with the same A/known-time key instead raise ValueError.
    # Explanation: groupby.indices supplies dealer positions within the slice; iloc selects A's rows and sorts them to 10:00,10:01, independently of B. Duplicate dealer timestamps cannot identify a unique event and are rejected.
    # Trick: iloc uses positions relative to the grouped slice; stable sorting preserves alignment, and duplicate-event ambiguity is rejected rather than silently resolved.
    states = []
    g = narrow.iloc[positions]
    for indices in g.groupby("firm", observed=True, sort=False).indices.values():
        h = g.iloc[indices]
        order = np.argsort(h._ns.to_numpy(), kind="stable")
        h = h.iloc[order]
        times = h._ns.to_numpy(dtype="int64")
        if len(times) > 1 and (np.diff(times) <= 0).any():
            raise ValueError("Event cache must have one event per firm/bond/side/known time")
        # CORE LOGIC: STEP 2 — Encode candidate-condition labels and pack dealer state arrays.
        # Input: A has three events with quantity_set=[('q=2.0',),('q=3.0',),('q=2.0',)],count=[1,1,2].
        # Output: quantity codes=[0,1,0],candidate_count=[1,1,2],aligned row by row with time and center.
        # Explanation: factorize assigns first-seen codes q=2 -> 0,q=3 -> 1,q=2 -> 0; retaining counts [1,1,2] separately lets the later guard detect changed candidate counts even when quantity matches.
        # Trick: Codes test equality of quantity sets, not quantity magnitude; zero and unknown-size candidates remain present.
        states.append((times, h.complete.fillna(False).to_numpy(dtype=bool),
                       h.center.to_numpy(dtype=float),
                       pd.factorize(h.quantity_set, sort=False)[0],
                       h.candidate_count.to_numpy(dtype=float)))
    return states


def _movement_at(states, times, day_ns, lookback_ns, age_ns, allow_exact):
    """One bond/side/day, all query times in a bounded block, one dealer per vote."""
    # CORE LOGIC: STEP 1 — Return zero support and NaN immediately when no dealer states exist.
    # Input: times=[10:30ns,10:35ns],states=[]
    # Output: Both rows have common_n/current_n/lookback_n/guarded_common_n=0,with direction and support ages all NaN.
    # Explanation: Neither query has an available dealer; each receives its own zero-count/NaN-direction row, without scanning, fabricating quotes or dropping queries.
    # Trick: This no-quote shortcut does not apply to an existing latest incomplete state; that state still selects the latest message.
    if not states:
        return _empty_move(len(times)), np.full(len(times), np.nan), np.full(len(times), np.nan)
    # CORE LOGIC: STEP 2 — Allocate query-by-dealer direction and support-age matrices.
    # Input: times has two queries; states has two dealers A and B.
    # Output: delta/guarded/ages/guarded_ages each have shape=(2,2),all NaN; current_n/old_n/changed_n=[0,0].
    # Explanation: Rows represent queries and fixed columns represent A/B; separate delta and guarded matrices receive changes only in the matching dealer column, never A's current value minus B's past value.
    # Trick: Fixed dealer columns preserve identity, preventing subtraction of centers from different dealer rosters.
    n = len(times)
    delta = np.full((n, len(states)), np.nan)
    guarded = delta.copy()
    ages = delta.copy()
    guarded_ages = delta.copy()
    current_n, old_n = np.zeros(n), np.zeros(n)
    changed_n = np.zeros(n)
    # CORE LOGIC: STEP 3 — Set the configured window start and require the same configured local day.
    # Input: query 3/3 00:10,lookback=30min,day_ns=3/3 00:00
    # Output: previous=March 2 23:40ns,same_day=False; allow_exact=True selects right.
    # Explanation: 00:10 minus 30 minutes is 23:40 on the previous day, before today's midnight, so same_day=False; a prior-day quote cannot form today's endpoint movement.
    # Trick: Elapsed windows use integer nanoseconds; a window crossing configured local midnight cannot borrow the previous day's endpoint.
    previous = times - lookback_ns
    same_day = previous >= day_ns
    side = "right" if allow_exact else "left"
    # CORE LOGIC: STEP 4 — Select each dealer's latest known-time event backward at both endpoints.
    # Input: A has complete 10:00 center100,complete 10:29 center105,incomplete 10:30; query10:30,lookback10:00.
    # Output: allow_exact=True selects now=10:30,old=10:00; False selects now=10:29 and old=-1.
    # Explanation: searchsorted(right)-1 includes the equal-time 10:30 incomplete message; left-1 excludes it and selects 10:29. At the 10:00 start, strict-before has no earlier message and returns -1.
    # Trick: right-1 means <= and left-1 means <; clamp -1 for safe array access, then mask it so the first message cannot be borrowed.
    for dealer, (stamp, complete, center, quantity, count) in enumerate(states):
        now = np.searchsorted(stamp, times, side=side) - 1
        old = np.searchsorted(stamp, previous, side=side) - 1
        ni, oi = np.maximum(now, 0), np.maximum(old, 0)
        now_age, old_age = times - stamp[ni], previous - stamp[oi]
        # CORE LOGIC: STEP 5 — Require latest complete, finite, fresh endpoints on the same day for common support.
        # Input: A has complete 10:00 center100 and incomplete 10:30; B has two complete endpoints with maximum age=30 minutes; age_min=30.
        # Output: A contributes current_n=0,lookback_n=1,but common=False; B supports both endpoints with common=True.
        # Explanation: A's current index points to incomplete 10:30, so valid_now=False without falling back to 10:29. B's endpoints are valid and exactly 30 minutes old; the <= boundary admits B as a common dealer.
        # Trick: Select the latest as-of event before checking completeness; never fall back to A's older complete message. The freshness threshold is inclusive.
        valid_now = (now >= 0) & complete[ni] & np.isfinite(center[ni]) & (now_age >= 0) & (now_age <= age_ns)
        valid_old = same_day & (old >= 0) & complete[oi] & np.isfinite(center[oi]) & (old_age >= 0) & (old_age <= age_ns)
        current_n += valid_now
        old_n += valid_old
        common = valid_now & valid_old
        # CORE LOGIC: STEP 6 — Preserve signed observed movement and flag quantity or count changes.
        # Input: A center100->104,quantity code0->1,count1->1; B center100->98,with unchanged quantity and count.
        # Output: delta=[4,-2],guarded=[NaN,-2],changed_n=1; each age is the larger endpoint message age.
        # Explanation: A changes by 104-100=4 but its quantity code changes, so only delta receives 4. B changes by 98-100=-2 under unchanged conditions, so both delta and guarded receive -2; one dealer changed conditions.
        # Trick: The guard does not delete quotes; it distinguishes movement under comparable conditions from observed differences after conditions change.
        changed = (quantity[ni] != quantity[oi]) | (count[ni] != count[oi])
        value = center[ni] - center[oi]
        delta[:, dealer] = np.where(common, value, np.nan)
        guarded[:, dealer] = np.where(common & ~changed, value, np.nan)
        age = np.maximum(now_age, old_age) / _MINUTE_NS
        ages[:, dealer] = np.where(common, age, np.nan)
        guarded_ages[:, dealer] = np.where(common & ~changed, age, np.nan)
        changed_n += common & changed
    # CORE LOGIC: STEP 7 — Aggregate common-dealer direction mean, median and guarded mean.
    # Input: delta=[4,-2],guarded=[NaN,-2]
    # Output: common_n=2,guarded_n=1,mean_bps=1,median_bps=1,guarded_mean_bps=-2
    # Explanation: Ordinary mean and median use 4 and -2, yielding (4-2)/2=1; guarded retains only -2 and divides by 1 rather than 2. Support counts are 2 and 1 respectively.
    # Trick: Observed and guarded means use their own actual support denominators; excluded values must not become zero votes.
    common_n = np.isfinite(delta).sum(axis=1)
    guarded_n = np.isfinite(guarded).sum(axis=1)
    result = _empty_move(n)
    result[:, _MOVE_COL["mean_bps"]] = _mean(delta, common_n)
    result[:, _MOVE_COL["median_bps"]] = _median(delta, common_n)
    result[:, _MOVE_COL["guarded_mean_bps"]] = _mean(guarded, guarded_n)
    # CORE LOGIC: STEP 8 — Report up/down/flat fractions and support counts on the common-dealer sample.
    # Input: delta=[4,-2,0],current_n=4,lookback_n=3; guarded has two valid values.
    # Output: up/down/flat each=1/3; common_n=3,current_n=4,lookback_n=3,guarded_common_n=2.
    # Explanation: [4,-2,0] has one increase, decrease and unchanged value, each divided by three common dealers. The current roster of four describes coverage, not the direction-fraction denominator.
    # Trick: Positive/negative means increase/decrease in the declared quote value, including price input; exactly zero is flat and unsupported fractions remain NaN.
    for name, values in [("up_fraction", delta > 0), ("down_fraction", delta < 0), ("flat_fraction", delta == 0)]:
        result[:, _MOVE_COL[name]] = np.divide(values.sum(axis=1), common_n,
            out=np.full(n, np.nan), where=common_n > 0)
    for name, values in [("common_n", common_n), ("current_n", current_n),
                         ("lookback_n", old_n), ("guarded_common_n", guarded_n)]:
        result[:, _MOVE_COL[name]] = values
    # CORE LOGIC: STEP 9 — Compute roster retention, condition-change rates and support ages.
    # Input: common_n=3,current_n=4,lookback_n=3,changed_n=1; dealer maximum endpoint ages=[0,5,10].
    # Output: retention=.75,condition_changed_fraction=1/3,mean_age=5,max_age=10
    # Explanation: Three common dealers / the larger roster of four = 0.75; one condition change / three common dealers = 1/3; endpoint-max ages [0,5,10] have mean 5 and maximum 10.
    # Trick: Retention uses the larger fresh roster as denominator; ages use each common dealer's older endpoint. Return guarded ages separately for issuer aggregation.
    result[:, _MOVE_COL["retention"]] = np.divide(common_n, np.maximum(current_n, old_n),
        out=np.full(n, np.nan), where=common_n > 0)
    result[:, _MOVE_COL["condition_changed_fraction"]] = np.divide(changed_n, common_n,
        out=np.full(n, np.nan), where=common_n > 0)
    result[:, _MOVE_COL["mean_support_age_min"]] = _mean(ages, common_n)
    result[:, _MOVE_COL["max_support_age_min"]] = _maximum(ages, common_n)
    return result, _mean(guarded_ages, guarded_n), _maximum(guarded_ages, guarded_n)


def _issuer_at(values, dealer_n, mean_age, max_age):
    """Target bond already excluded; each finite donor bond supplies one vote."""
    # CORE LOGIC: STEP 1 — Count finite directions from other bonds, keeping dealer counts as support information.
    # Input: Target bond already excluded; values has one row=[2,6,NaN],dealer_n=[3,1,5].
    # Output: other_bond_n=2,common_dealer_n=4;supported=True
    # Explanation: The finite mask retains directions 2 and 6; the third bond has no direction and cannot vote despite five recorded dealers. Valid bonds provide 3+1=4 dealer-bond supports.
    # Trick: The unsupported bond's five dealers do not count; each valid bond always supplies one vote.
    valid = np.isfinite(values)
    n = valid.sum(axis=1)
    result = np.full((len(values), len(_ISSUER_SUFFIXES)), np.nan)
    result[:, 5], result[:, 6] = n, np.where(valid, dealer_n, 0).sum(axis=1)
    supported = n >= 2
    # CORE LOGIC: STEP 2 — Compute direction and sign fractions only with at least two other bonds.
    # Input: values=[2,6,NaN]; a second row has only [2,NaN,NaN].
    # Output: First row mean=4,median=4,up=1,down=0,flat=0; second row mean/median/up/down/flat/dispersion/mean_age/max_age all NaN,but other_bond_n=1 and common_dealer_n retains that bond's actual support.
    # Explanation: Votes 2 and 6 have equal-bond mean and median 4, and both are positive, so up=1. A single vote misses the predefined two-bond minimum; direction stays NaN while the count remains 1.
    # Trick: Do not weight by dealer_n or quantity; min2 is a predefined support requirement.
    if supported.any():
        v, nn = values[supported], n[supported]
        avg = _mean(v, nn)
        result[supported, 0] = avg
        result[supported, 1] = _median(v, nn)
        for col, flag in [(2, v > 0), (3, v < 0), (4, v == 0)]:
            result[supported, col] = flag.sum(axis=1) / nn
        # CORE LOGIC: STEP 3 — Compute equal-bond dispersion and support ages.
        # Input: Two donors have directions 2/6,guarded-dealer mean ages 1/5,and maximum ages 2/8.
        # Output: dispersion=2,mean_support_age=3,max_support_age=8
        # Explanation: The mean vote is 4; squared deviations 4 and 4 average to 4, whose square root is 2. Bond mean ages average to (1+5)/2=3 and the maximum age is max(2,8)=8.
        # Trick: Standard deviation uses the bond count with ddof=0; average ages first within each bond, then equally across bonds, preventing dealer-rich bonds from dominating.
        result[supported, 7] = np.sqrt(np.nansum((v - avg[:, None]) ** 2, axis=1) / nn)
        result[supported, 8] = _mean(np.where(valid[supported], mean_age[supported], np.nan), nn)
        result[supported, 9] = _maximum(np.where(valid[supported], max_age[supported], np.nan), nn)
    return result


def _dictionary(include_issuer):
    # SETUP LOGIC: Define fixed direction-feature descriptions without computing observed data.
    definitions = {
        "mean_bps": "Equal-dealer mean of current minus lookback event candidate medians",
        "median_bps": "Median of common-dealer current minus lookback candidate medians",
        "guarded_mean_bps": "Equal-dealer mean change where quantity_set and candidate_count match at endpoints",
        "up_fraction": "Fraction of supported votes with a strictly positive quote-value change",
        "down_fraction": "Fraction of supported votes with a strictly negative quote-value change",
        "flat_fraction": "Fraction of supported votes with exactly zero quote-value change",
        "common_n": "Dealer count complete and fresh at both endpoints",
        "current_n": "Complete fresh dealer count at query time",
        # SETUP LOGIC: Define support, retention and age descriptions; preserve existing strings.
        "lookback_n": "Complete fresh dealer count at window start on the same configured local day",
        "guarded_common_n": "Common dealer count with unchanged quantity_set and candidate_count",
        "retention": "common_n / max(current_n, lookback_n); unknown without any common dealer",
        "condition_changed_fraction": "Common dealers with changed quantity_set or candidate_count / common_n",
        "mean_support_age_min": "Mean of the larger of the two endpoint message ages",
        "max_support_age_min": "Maximum message age across both endpoints and supported votes",
    }
    # SETUP LOGIC: Attach units and fixed support descriptions to each side and field.
    result = {}
    for side in _SIDES:
        for name in _MOVE_SUFFIXES:
            unit = "count" if name in _COUNT_SUFFIXES else "minutes" if name.endswith("_min") else "quote value unit" if name.endswith("_bps") else "fraction"
            result[f"bcq_{side}_move_{name}"] = {"unit": unit, "definition": definitions[name],
                "support": "Same bond/side/dealer, latest complete fresh events, same configured local day; no automatic quote/quantity filtering. Legacy _bps names follow declared quote-value units."}
        # SETUP LOGIC: Describe issuer features with explicit min2 and target-bond exclusion rules.
        if include_issuer:
            for name in _ISSUER_SUFFIXES:
                unit = "count" if name in _COUNT_SUFFIXES else "minutes" if name.endswith("_min") else "quote value unit" if name.endswith("_bps") else "fraction"
                definition = ({"other_bond_n": "Number of other bonds with a guarded common-dealer direction",
                    "common_dealer_n": "Sum of guarded common-dealer support across other bonds; not a price weight",
                    "dispersion_bps": "Population standard deviation of other-bond mean directions",
                    "mean_bps": "Equal-bond mean of other bonds' guarded common-dealer mean directions",
                    "median_bps": "Median of other bonds' guarded common-dealer mean directions",
                    # SETUP LOGIC: Define issuer-age descriptions as metadata, not direction calculations.
                    "mean_support_age_min": "Equal-bond mean of donor guarded-dealer mean maximum endpoint ages",
                    "max_support_age_min": "Maximum endpoint age among donor guarded dealers"}.get(name, definitions.get(name, name)))
                # SETUP LOGIC: Assemble issuer dictionary entries and return metadata.
                result[f"bcq_{side}_issuer_move_{name}"] = {"unit": unit, "definition": definition,
                    "support": "Target CUSIP excluded; one bond per vote; >=2 other bonds for non-count values; prefix-known stable issuer at window start through query"}
    return result


def _explicit_movement_queries(frame, required, positions):
    # VALIDATION LOGIC: Explicit final-evaluation positions never relabel Test or permit ambiguous row identities.
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError('query_positions must be a one-dimensional integer position array')
    if len(np.unique(positions)) != len(positions) or (positions < 0).any() or (positions >= len(frame)).any():
        raise ValueError('query_positions must be unique valid positions in the original frame')
    # CORE LOGIC: STEP 1 — Select explicit queries while recording which original rows were requested.
    # Input: query_positions=None, frame row_id=[7,8,9],split=['Train','Outside experiment','Test']; positions=[2,1].
    # Output: query row_id=[9,8] with original splits ['Test','Outside experiment']; allowed=[False,True,True].
    # Explanation: iloc selects original positions in caller order; the boolean mask only counts excluded rows.
    # Trick: No split label is changed, and duplicate times remain separate unique row_id queries.
    queries = frame.iloc[positions][list(required)].copy().reset_index(drop=True)
    allowed = pd.Series(False, index=frame.index)
    allowed.iloc[positions] = True
    return queries, allowed


def _full_issuer_history(frame):
    # SETUP LOGIC: Prepare known-time issuer observations from the complete unchanged frame, without target values.
    history = frame[['cusip', 'time', 'ISSUER']].copy()
    history['cusip'] = history.cusip.astype('string')
    history['_ns'] = aware_time(history.time).array.as_unit('ns').asi8
    return _issuer_records(history)


def build_movement_features(frame, event_cache, *, lookback_min=30, age_min=30,
                            allow_exact=True, include_issuer=True, progress=None, query_positions=None):
    """Return (feature frame, metadata) from cached events; default queries remain Train/Validation.

    Query order/row_id are retained; Test and other split rows are explicitly
    excluded unless query_positions explicitly requests original frame positions.
    Explicit queries use full-frame issuer observations only at their known times;
    future labels never backfill an earlier query. Latest incomplete messages block old valid state. Exact matches
    apply to quote known-times only. Default issuer mapping uses Train/Validation
    observations known by the query; donors must already be reliably mapped at
    the window start and remain so through the query. Quote-level medians are
    never pooled across bonds. The lookback window is explicit in minutes.

    Each event is indexed once. Dealer arrays are reused within an local day, and
    issuer query blocks share all donor asof calculations before excluding each
    target CUSIP. Memory for issuer query matrices is bounded to 512 times per
    block. No raw input, event aggregation, model fit or disk cache build occurs.
    """
    # SETUP LOGIC: Start timing while preserving the existing function description and signature.
    started = perf_counter()
    # VALIDATION LOGIC: Check declared windows and the existing event cache; never rebuild raw events automatically.
    if not np.isfinite(lookback_min) or lookback_min < 0 or not np.isfinite(age_min) or age_min <= 0:
        raise ValueError("lookback_min must be nonnegative finite and age_min positive finite")
    required = {"row_id", "cusip", "time", "split"} | ({"ISSUER"} if include_issuer else set())
    if not required.issubset(frame.columns):
        raise ValueError("Movement frame missing columns: " + ", ".join(sorted(required - set(frame.columns))))
    if not isinstance(event_cache, dict) or not isinstance(event_cache.get("events"), pd.DataFrame):
        raise ValueError("Provide the normalized event cache; movement does not build events")
    # CORE LOGIC: STEP 1 — Validate event schema and select the default Train/Validation queries.
    # Input: query_positions=None, frame row_id=[7,8,9],split=['Train','Validation','Test'],with all required event columns present.
    # Output: q retains row_id7/8 and excludes Test9; nonunique row_id or missing event fields raise an error.
    # Explanation: isin yields [True,True,False], so loc retains only 7 and 8 before reset_index creates positional indices. Test row 9 neither gets features nor establishes issuer affiliation early.
    # Trick: Default queries exclude Test from both features and issuer mappings; the explicit-position branch below has separate known-time rules.
    events = event_cache["events"]
    event_columns = {"firm", "cusip", "side", "quote_timestamp_ET", "day", "complete", "center", "quantity_set", "candidate_count"}
    if not event_columns.issubset(events.columns):
        raise ValueError("Movement event cache missing columns: " + ", ".join(sorted(event_columns - set(events.columns))))
    allowed = frame.split.isin(["Train", "Validation"])
    q = frame.loc[allowed, list(required)].copy().reset_index(drop=True)
    # ORCHESTRATION LOGIC: An explicit final scope selects original positions without changing default research behavior.
    if query_positions is not None:
        q, allowed = _explicit_movement_queries(frame, required, query_positions)
    # VALIDATION LOGIC: Every requested target keeps one unique original identity.
    if q.row_id.isna().any() or q.row_id.duplicated().any():
        raise ValueError("Train/Validation row_id must be nonmissing and unique")
    # SETUP LOGIC: Normalize time storage, retain original CUSIP keys and prepare issuer records without backfilling historical labels.
    q["time"] = aware_time(q.time)
    # Preserve normalized exact CUSIP keys; do not silently transform another
    # identifier into a quoted bond or change identity joins.
    q["cusip"] = q.cusip.astype("string")
    q["_ns"] = q.time.array.as_unit("ns").asi8
    q["_day"] = q.time.dt.normalize().array.as_unit("ns").asi8
    lookback_ns, age_ns = int(lookback_min * _MINUTE_NS), int(age_min * _MINUTE_NS)
    records = (_issuer_records(q) if query_positions is None else _full_issuer_history(frame)) if include_issuer else {}
    q["_issuer"] = None
    # CORE LOGIC: STEP 2 — Map target queries to issuer affiliations reliably known at that time.
    # Input: X record=('I',10:00ns,11:00ns);queries09:59/10:30/11:00
    # Output: _issuer=[None,'I',None]
    # Explanation: For X's three query positions, 09:59 is before first=10:00,10:30 lies in [10:00,11:00),and 11:00 is excluded at the conflict boundary; assign I only to the middle row.
    # Trick: groupby.indices refers to positions after reset_index; prefix-valid masks prevent future conflicts from changing earlier results.
    for bond, positions in q.groupby("cusip", observed=True, sort=False).indices.items():
        record = records.get(bond)
        if record is not None:
            label, first, conflict = record
            positions = np.asarray(positions)
            valid = (q._ns.to_numpy()[positions] >= first) & (q._ns.to_numpy()[positions] < conflict)
            q.loc[positions[valid], "_issuer"] = label
    # PROGRESS LOGIC: Report preparation of the narrow event index without raw aggregation.
    if progress is not None:
        progress("movement_index", None, None, f"Indexing existing events for {len(q):,} requested queries; no raw aggregation")
    # CACHEING LOGIC: Reuse existing narrow events without triggering raw aggregation; _event_index defines filtering.
    narrow, index, available_days = _event_index(events, q, set(q.cusip.dropna()) | set(records), lookback_ns, age_ns)
    # SETUP LOGIC: Record completion time for event indexing.
    indexed = perf_counter()
    # CORE LOGIC: STEP 3 — Initialize declared features with zero unsupported counts.
    # Input: q=[{row_id:7,cusip:'X',time:'10:00'},{row_id:8,cusip:'X',time:'10:30'}],include_issuer=True
    # Output: output shape=(2,48),with identical rows: bid/ask move_common_n/current_n/lookback_n/guarded_common_n and issuer_move_other_bond_n/common_dealer_n give 12 zero columns; the remaining 36 columns are NaN.
    # Explanation: Allocate 48 columns for each target; set only the 12 explicitly named count columns to 0 and leave the other 36 NaN. Unsupported targets remain present without fabricated zero direction.
    # Trick: Coverage plots must test finite guarded_mean/issuer_mean values; a non-null zero support count is not a supported feature.
    names = DIRECTION_FEATURES + (ISSUER_FEATURES if include_issuer else [])
    output = np.full((len(q), len(names)), np.nan)
    for i, name in enumerate(names):
        if any(name.endswith("_" + suffix) for suffix in _COUNT_SUFFIXES):
            output[:, i] = 0
    # CORE LOGIC: STEP 4 — Index candidate donor bonds once by known issuer.
    # Input: records={'X':('I',10:00ns,11:00ns),'Y':('I',09:00ns,maxint)}
    # Output: issuer_bonds={'I':['X','Y']}
    # Explanation: Each X/Y record has label I, so append both bonds under dictionary key I, yielding [X,Y]. This coarse index alone does not establish validity for a particular query.
    # Trick: This step only builds an index; each query still checks first/conflict bounds, so future affiliation is not made available early.
    issuer_bonds = {}
    for bond, record in records.items():
        issuer_bonds.setdefault(record[0], []).append(bond)
    # Unknown mappings require only their own bond/day; they never cause an
    # issuer-wide donor scan. Invalid-time rows retain the empty schema.
    # CORE LOGIC: STEP 5 — Share issuer/day queries for reliable affiliations; otherwise group by own bond/day.
    # Input: q positions0/1/2/3 are X/I/March2 10:30,Y/I/March2 10:30,Z/None/March2 10:30,X/I/NaT.
    # Output: groups=[(March2 ET midnight ns,'I',array([0,1])),(same midnight ns,None,array([2]))]; position3 is ungrouped with zero support and NaN direction retained.
    # Explanation: known puts positions0/1 in the same I/day group; Z's unknown label groups it by its own bond. NaT row3 skips computation but remains in the initialized output.
    # Trick: Unknown affiliation must not trigger an all-issuer scan; sort groups only by ET date so state caches can be released between days.
    valid = q._ns.ne(np.iinfo(np.int64).min) & q.cusip.notna() & q.cusip.str.strip().ne("")
    known = valid & q._issuer.notna() if include_issuer else pd.Series(False, index=q.index)
    groups = []
    for (day, label), positions in q.loc[known].groupby(["_day", "_issuer"], observed=True, sort=False).groups.items():
        groups.append((int(day), label, np.asarray(positions)))
    for (day, bond), positions in q.loc[valid & ~known].groupby(["_day", "cusip"], observed=True, sort=False).groups.items():
        groups.append((int(day), None, np.asarray(positions)))
    groups.sort(key=lambda item: item[0])
    # CACHEING LOGIC: Initialize a single-ET-day dealer-array cache; clear it on day changes rather than retaining a huge all-query matrix.
    cached_day, states_cache = None, {}
    # CORE LOGIC: STEP 6 — Stably sort same-day queries and deduplicate into blocks of 512 times.
    # Input: Within I/March2,positions=[row7@10:35,row8@10:30,row9@10:30].
    # Output: Sorted rows=[8,9,7],unique=[10:30ns,10:35ns]; each timestamp is computed once.
    # Explanation: Sort both 10:30 rows before 10:35; stable ordering preserves 8 before 9. np.unique collapses 10:30 into one calculation point, later mapped back to both trades.
    # Trick: Reuse states_cache within the day and clear it on date changes; an unavailable day retains empty outputs directly.
    for done, (day, label, positions) in enumerate(groups, 1):
        if day != cached_day:
            cached_day, states_cache = day, {}
        if day in available_days:
            order = np.argsort(q._ns.to_numpy()[positions], kind="stable")
            positions = positions[order]
            ns = q._ns.to_numpy()[positions]
            unique = np.unique(ns)
            for start in range(0, len(unique), _QUERY_BLOCK):
                times = unique[start:start + _QUERY_BLOCK]
                # PROGRESS LOGIC: Show each block's time range before processing its 512 unique queries, keeping large-issuer progress visible.
                if progress is not None:
                    progress("movement_features", done - 1, len(groups),
                        f"Query group {done:,}/{len(groups):,}: {label or 'unmapped bond'}; unique times {start + 1:,}–{start + len(times):,}/{len(unique):,}")
                # CORE LOGIC: STEP 7 — Map repeated targets into the current time block and select potentially known donors.
                # Input: Sorted ns=[10:30,10:30,10:35],with both unique times in the block; X/Y are targets and Z's I affiliation was known at 10:00.
                # Output: rows contains all three records,row_times=[0,0,1]; bonds is the union X/Y/Z,with fixed columns in bond_index.
                # Explanation: left finds the first 10:30 at position0 and right passes the final 10:35 to position3, preserving all rows; their unique-time indices are 0,0,1.
                # Trick: left retains the first duplicate and right includes the last duplicate; donor prefiltering still needs per-time prefix checks, never backfilling early queries from block-end information.
                a = np.searchsorted(ns, times[0], side="left")
                b = np.searchsorted(ns, times[-1], side="right")
                rows = positions[a:b]
                row_times = np.searchsorted(times, q._ns.to_numpy()[rows])
                targets = q.cusip.to_numpy()[rows]
                donors = [] if label is None else [bond for bond in issuer_bonds.get(label, [])
                    if records[bond][1] <= times[-1] - lookback_ns and records[bond][2] > times[0]]
                bonds = sorted(set(donors) | set(targets))
                bond_index = {bond: i for i, bond in enumerate(bonds)}
                # CORE LOGIC: STEP 8 — Allocate donor-direction matrices and skip absent bond/side/day combinations.
                # Input: times=[10:30ns,10:35ns],bonds=['X','Y','Z'],side='bid'; index contains only (X,bid,today) and (Y,bid,today).
                # Output: Initial donor_values/donor_age/donor_max_age=[[NaN,NaN,NaN],[NaN,NaN,NaN]],donor_n=[[0,0,0],[0,0,0]]; Z skips as-of computation and retains NaN direction with zero support.
                # Explanation: Two times by three bonds form 2x3 matrices; only indexed X/Y use as-of calculations. Z never scans dealers and retains NaN direction and zero support.
                # Trick: Matrix columns always identify bonds; process bid/ask separately so donor votes never mix sides.
                for side_i, side in enumerate(_SIDES):
                    donor_values = np.full((len(times), len(bonds)), np.nan)
                    donor_n = np.zeros_like(donor_values)
                    donor_age = donor_values.copy()
                    donor_max_age = donor_values.copy()
                    for bond_i, bond in enumerate(bonds):
                        key = (bond, side, day)
                        if key not in index:
                            continue
                        # CACHEING LOGIC: Reuse existing bond/side/day arrays; only the first request extracts dealer states from the narrow index.
                        if key not in states_cache:
                            states_cache[key] = _dealer_states(narrow, index[key])
                        # CORE LOGIC: STEP 9 — Compute common-dealer direction for the current bond and write its own query features.
                        # Input: X's common-dealer direction is +4; rows has two X queries at 10:30 and one Y query at 10:35.
                        # Output: Both X rows receive mean=4,guarded_mean=4; X's results do not overwrite the Y row.
                        # Explanation: Shared calculation yields X's +4 at 10:30; own identifies both X targets and row_times selects the same computed result twice. The Y position is outside own and remains untouched.
                        # Trick: own selects positions by CUSIP while row_times maps the shared computation axis; computational reuse must preserve duplicate trade rows.
                        movement, mean_age, max_age = _movement_at(states_cache[key], times, day, lookback_ns, age_ns, allow_exact)
                        own = np.flatnonzero(targets == bond)
                        if len(own):
                            offset = side_i * len(_MOVE_SUFFIXES)
                            output[rows[own], offset:offset + len(_MOVE_SUFFIXES)] = movement[row_times[own]]
                        # CORE LOGIC: STEP 10 — Require donor affiliation known at window start and unconflicted through query time.
                        # Input: Y's I label is first known at 10:01; queries 10:30/10:35 have window starts 10:00/10:05.
                        # Output: mapped=[False,True]; Y's first vote=NaN,the second uses its guarded_mean and guarded dealer support.
                        # Explanation: The 10:30 query starts at 10:00, before Y's known time 10:01, so exclude its vote. At10:35, the 10:05 start already knows I and no conflict has occurred, allowing Y's direction.
                        # Trick: A label known only at window end cannot fill unknown start affiliation; conflict must be strictly later than the query.
                        record = records.get(bond) if label is not None else None
                        if record is not None and record[0] == label:
                            mapped = (record[1] <= times - lookback_ns) & (record[2] > times)
                            donor_values[:, bond_i] = np.where(mapped, movement[:, _MOVE_COL["guarded_mean_bps"]], np.nan)
                            donor_n[:, bond_i] = movement[:, _MOVE_COL["guarded_common_n"]]
                            donor_age[:, bond_i], donor_max_age[:, bond_i] = mean_age, max_age
                    # CORE LOGIC: STEP 11 — Exclude each target's own bond completely, then aggregate issuer direction and support.
                    # Input: donor_values@10:30=[X100,Y2,Z6],target=X,Y/Z dealer_n=3/1
                    # Output: After exclusion,[NaN,2,6]; issuer mean=4,other_bond_n=2,common_dealer_n=4,written to X's row.
                    # Explanation: Copy [100,2,6] for target X and replace X's column with NaN; Y/Z contribute equally to (2+6)/2=4,with two bonds and 3+1=4 dealer-bond supports. X's 100 cannot affect the result.
                    # Trick: copy prevents X's exclusion from mutating the shared matrix used for Y targets; exclude self from means, medians, ages, dispersion and support counts alike.
                    if label is not None:
                        # Share donor snapshots, then remove the target bond from
                        # every vote, count, dispersion and support-age statistic.
                        for bond in set(targets):
                            own = np.flatnonzero(targets == bond)
                            ti = row_times[own]
                            values = donor_values[ti].copy()
                            values[:, bond_index[bond]] = np.nan
                            issuer = _issuer_at(values, donor_n[ti], donor_age[ti], donor_max_age[ti])
                            offset = len(DIRECTION_FEATURES) + side_i * len(_ISSUER_SUFFIXES)
                            output[rows[own], offset:offset + len(_ISSUER_SUFFIXES)] = issuer
        # PROGRESS LOGIC: Report completion by query-group count without changing feature results.
        if progress is not None and (done == len(groups) or done % max(1, len(groups) // 100) == 0):
            progress("movement_features", done, len(groups), f"Completed {done:,}/{len(groups):,} issuer/day or unmapped bond/day query blocks")
    # SETUP LOGIC: Record completion time for direction calculations.
    computed = perf_counter()
    # CORE LOGIC: STEP 12 — Attach additional features to the original selected query identities.
    # Input: include_issuer=False,q=[{row_id:8,cusip:'X',time:'10:30'},{row_id:7,cusip:'Y',time:'10:30'}]; in the 14 _MOVE_SUFFIXES order,bid vectors are [4,4,4,1,0,0,1,1,1,1,0,0,0,1] and [-2,-2,-2,0,1,0,1,1,1,1,0,0,0,1]; both ask vectors are [NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,NaN,NaN,0].
    # Output: A 2-row, 31-column DataFrame; first three columns are (8,'X','10:30') then (7,'Y','10:30'); each row's remaining 28 columns concatenate its explicit 14-value bid and 14-value ask vectors, without sorting by row_id.
    # Explanation: q and the numeric matrix share a RangeIndex, so concat(axis=1) joins by position; row_id 8 stays first and 7 second. Merging on the repeated 10:30 timestamp could instead create many-to-many duplicates.
    # Trick: Reset both q and output to RangeIndex; concat(axis=1) aligns the index rather than time, preventing duplicate-time cross matches.
    result = pd.concat([q[["row_id", "cusip", "time"]], pd.DataFrame(output, columns=names)], axis=1)
    # SETUP LOGIC: Record the actual window, local day, as-of, condition guard, issuer-causality and min2 rules.
    config = dict(lookback_min=float(lookback_min), age_min=float(age_min), allow_exact=bool(allow_exact),
        include_issuer=bool(include_issuer), same_local_day=True, timezone=str(q.time.dt.tz),
        direction_support="same dealer; latest complete and fresh at both endpoints",
        condition_guard="quantity_set and candidate_count unchanged at endpoints",
        retention_denominator="max(current_n, lookback_n)", issuer_min_other_bonds=2,
        issuer_vote="one bond; guarded common-dealer mean change",
        issuer_mapping_source="Train/Validation frame ISSUER prefix only",
        issuer_mapping_rule="target known at query; donor known at window start and no observed conflict through query; unknown/conflicted excluded")
    # CONFIGURATION LOGIC: Separate explicit final scope provenance from the unchanged validation-sidecar protocol.
    if query_positions is not None:
        config['issuer_mapping_source'] = 'Full original frame ISSUER known-time prefix; explicit query positions'
    # CACHEING LOGIC: A local full-precision fingerprint records input provenance without importing training.
    from .cache import fingerprint as frame_fingerprint
    # CACHEING LOGIC: Record frame/cache identities, fixed settings, the dictionary and timings for save/restore validation.
    fingerprint = frame_fingerprint(frame) if len(q) else None
    result.attrs.update(movement_config=config, event_cache_key=event_cache.get("cache_key"),
        source_frame_sha256=fingerprint, feature_dictionary=_dictionary(include_issuer),
        excluded_rows=int((~allowed).sum()), included_splits=(["Train", "Validation"] if query_positions is None else
            frame.iloc[np.asarray(query_positions)]['split'].drop_duplicates().tolist()),
        timings=dict(index_s=indexed - started, features_s=computed - indexed,
                     fingerprint_s=perf_counter() - computed, total_s=perf_counter() - started))
    # PROGRESS LOGIC: Report sidecar completion and the actual default or explicit query scope.
    if progress is not None:
        scope_note = 'Test excluded' if query_positions is None else 'explicit original query scope'
        progress("movement_features", len(groups), len(groups), f"Movement sidecar ready: {len(result):,} rows; {scope_note}")
    # OUTPUT LOGIC: Return a separate sidecar without mutating the original frame, fitting models or reading targets.
    return result, dict(result.attrs)
