"""Reusable observed-date folds with explicit training windows, holdouts and label-availability gates."""
# SETUP LOGIC: The date splitter depends only on pandas/numpy, not on an estimator library.
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .data import local_time, read_data


@dataclass(frozen=True)
class WalkForwardConfig:
    """Count observed local dates, retaining complete, nonoverlapping validation blocks.

    ``max_train_dates=None`` expands training; a positive cap creates a rolling window.
    ``n_splits`` retains the latest requested complete folds. Final holdout dates and
    their optional preceding embargo never enter any training or validation fold.
    """
    # CONFIGURATION LOGIC: Fold boundaries are declared before examining model errors.
    min_train_dates: int = 20
    validation_dates: int = 5
    embargo_dates: int = 0
    step_dates: int = None
    max_train_dates: int = None
    n_splits: int = None
    holdout_dates: int = 0
    holdout_embargo_dates: int = 0
    label_available_column: str = None

    def __post_init__(self):
        # VALIDATION LOGIC: Reject overlapping validation windows and impossible integer settings early.
        positive = ['min_train_dates', 'validation_dates', 'step_dates', 'max_train_dates', 'n_splits']
        for name in positive + ['embargo_dates', 'holdout_dates', 'holdout_embargo_dates']:
            value = getattr(self, name)
            if value is None and name in {'step_dates', 'max_train_dates', 'n_splits'}:
                continue
            minimum = 1 if name in positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if self.step_dates is not None and self.step_dates < self.validation_dates:
            raise ValueError('step_dates must be >= validation_dates; each validation record may be scored only once.')
        if self.max_train_dates is not None and self.max_train_dates < self.min_train_dates:
            raise ValueError('max_train_dates must be >= min_train_dates.')
        if self.holdout_embargo_dates and not self.holdout_dates:
            raise ValueError('holdout_embargo_dates requires a positive holdout_dates reservation.')
        if self.label_available_column is not None and (not isinstance(self.label_available_column, str) or not self.label_available_column.strip()):
            raise ValueError('label_available_column must be a nonempty source column name or None.')

    def to_dict(self):
        # SERIALIZATION LOGIC: Publish numerical policy without model objects or source records.
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardFold:
    # CONFIGURATION LOGIC: Positions refer to the original supplied frame, including an unsorted source index.
    fold_id: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    train_dates: tuple
    validation_dates: tuple
    label_purged_rows: int = 0

    def to_dict(self):
        # REPORTING LOGIC: Compact exact date boundaries and counts are sufficient for a fold audit table.
        return dict(fold_id=self.fold_id, train_start=self.train_dates[0], train_end=self.train_dates[-1],
                    validation_start=self.validation_dates[0], validation_end=self.validation_dates[-1],
                    train_dates=len(self.train_dates), validation_dates=len(self.validation_dates),
                    train_rows=len(self.train_positions), validation_rows=len(self.validation_positions),
                    label_purged_rows=self.label_purged_rows,
                    scheduled_train_rows=len(self.train_positions)+self.label_purged_rows)


def walk_forward_splits(data, time_column, settings=None, *, timezone='UTC'):
    """Build positional folds without accessing targets, features, or model predictions."""
    # VALIDATION LOGIC: Unlike descriptive date plots, fold assignment cannot tolerate an unknown timestamp.
    settings = settings if isinstance(settings, WalkForwardConfig) else WalkForwardConfig(**(settings or {}))
    frame = data if isinstance(data, pd.DataFrame) else read_data(data)
    if frame.columns.duplicated().any() or time_column not in frame:
        raise ValueError('Provide unique source columns and an existing time_column.')
    times = local_time(frame[time_column], timezone)
    if times.isna().any():
        raise ValueError('Walk-forward dates require valid timestamps; resolve missing/ambiguous times upstream.')
    label_times = None
    if settings.label_available_column is not None:
        if settings.label_available_column not in frame:
            raise ValueError(f'Missing label availability column: {settings.label_available_column!r}')
        label_times = local_time(frame[settings.label_available_column], timezone)
    # CORE LOGIC: STEP 1 — Remove the reserved final date block and its preceding buffer from development.
    # Input: dates=pd.date_range('2026-01-01', periods=6); holdout_dates=2; holdout_embargo_dates=1.
    # Output: development=['2026-01-01','2026-01-02','2026-01-03']; buffer='2026-01-04'; holdout=['2026-01-05','2026-01-06'].
    # Explanation: Reservations count observed local dates, not calendar days or row counts.
    # Trick: Timestamp normalization groups every same-local-date record together, including unequal daily row counts.
    days = times.dt.normalize()
    dates = pd.Index(days.drop_duplicates()).sort_values()
    reserved = settings.holdout_dates + settings.holdout_embargo_dates
    development = dates[:len(dates)-reserved] if reserved else dates
    if reserved >= len(dates):
        development = dates[:0]
    # CORE LOGIC: STEP 2 — Enumerate complete forward validation blocks after the minimum train and embargo.
    # Input: development=pd.date_range('2026-01-01', periods=10); min_train_dates=3; validation_dates=2; embargo_dates=1.
    # Output: validation=[['2026-01-05','2026-01-06'],['2026-01-07','2026-01-08'],['2026-01-09','2026-01-10']].
    # Explanation: A default step of two creates disjoint validation rows; a partial final block is unscored.
    # Trick: n_splits selects the latest chronological folds using dates alone, never their measured performance.
    step = settings.step_dates or settings.validation_dates
    first = settings.min_train_dates + settings.embargo_dates
    starts = list(range(first, len(development)-settings.validation_dates+1, step))
    if settings.n_splits is not None:
        starts = starts[-settings.n_splits:]
    if not starts:
        raise ValueError('Too few development dates for one complete training/embargo/validation fold.')
    # ORCHESTRATION LOGIC: Materialize a compact reusable plan; no model is fitted and no holdout label is read.
    return [_make_fold(days, development, start, index+1, settings, label_times) for index, start in enumerate(starts)]


def _make_fold(days, dates, start, fold_id, settings, label_times=None):
    # CORE LOGIC: STEP 1 — Use only dates strictly before this fold's embargo and validation window.
    # Input: dates=pd.date_range('2026-01-01', periods=10); start=6; embargo_dates=1; max_train_dates=3; validation_dates=2.
    # Output: train=['2026-01-03','2026-01-04','2026-01-05']; embargo='2026-01-06'; validation=['2026-01-07','2026-01-08'].
    # Explanation: Expanding mode starts at January 1; rolling mode retains only the latest allowed training dates.
    # Trick: Boundary arithmetic uses half-open date-position slices, so the embargo contains no fitted labels.
    train_stop = start-settings.embargo_dates
    train_start = max(0, train_stop-settings.max_train_dates) if settings.max_train_dates else 0
    train_dates = dates[train_start:train_stop]
    validation_dates = dates[start:start+settings.validation_dates]
    # CORE LOGIC: STEP 2 — Return original row positions while retaining chronological date metadata.
    # Input: source days=['2026-01-08','2026-01-03','2026-01-07','2026-01-05','2026-01-06']; train=['2026-01-03','2026-01-04','2026-01-05']; validation=['2026-01-07','2026-01-08'].
    # Output: train_positions=[1,3], validation_positions=[0,2]; original source order is preserved.
    # Explanation: Both models index exactly these same rows with iloc, independent of source index labels.
    # Trick: Positional arrays remain unambiguous even when a caller's pandas index has repeated labels.
    train_positions = np.flatnonzero(days.isin(train_dates).to_numpy())
    validation_positions = np.flatnonzero(days.isin(validation_dates).to_numpy())
    # CORE LOGIC: STEP 3 — Purge labels not demonstrably available before this validation window begins.
    # Input: train_positions=[1,3,5], label times=['2026-01-06 23:59','2026-01-07 00:00',NaT], validation starts '2026-01-07 00:00'.
    # Output: train_positions=[1], label_purged_rows=2; the row known exactly at the boundary is excluded.
    # Explanation: Labels may arrive after their feature timestamps; missing availability cannot establish causal training.
    # Trick: Positional masking preserves source order; date metadata describes the scheduled window before this purge.
    purged = 0
    if label_times is not None:
        known = label_times.iloc[train_positions]
        available = known.notna() & known.lt(validation_dates[0])
        purged = int((~available).sum())
        train_positions = train_positions[available.to_numpy()]
    return WalkForwardFold(fold_id, train_positions, validation_positions,
                           tuple(train_dates.strftime('%Y-%m-%d')), tuple(validation_dates.strftime('%Y-%m-%d')), purged)
