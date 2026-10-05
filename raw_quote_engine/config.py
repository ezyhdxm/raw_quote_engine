"""Explicit source schemas, units, and bounded quote-feature settings."""
# SETUP LOGIC: Configuration can be imported without reading data or loading a model.
from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class TransactionColumns:
    """Source names; optional metadata is used only when its name is supplied."""
    # CONFIGURATION LOGIC: Original columns remain available for arbitrary baseline features and slices.
    bond: str
    time: str
    target: str
    id: str = None
    issuer: str = None
    sector: str = None
    quantity: str = None
    prev_quantity: str = None
    maturity_date: str = None
    maturity_years: str = None
    anchor: str = None
    cpp: str = None
    split: str = None
    actual: str = None
    rollover_adjustment: str = None

    def __post_init__(self):
        # VALIDATION LOGIC: Required mappings must be nonempty column names, not positional indices.
        _column_names(self, ('bond', 'time', 'target'))


@dataclass(frozen=True)
class QuoteColumns:
    """Choose wide bid/ask columns or a long side/value pair; known_time is mandatory."""
    # CONFIGURATION LOGIC: An exchange/original timestamp never substitutes for the known timestamp.
    bond: str
    known_time: str
    dealer: str
    bid: str = None
    ask: str = None
    side: str = None
    value: str = None
    quantity: str = None
    bid_size: str = None
    ask_size: str = None
    original_timestamp: str = None
    source_id: str = None

    def __post_init__(self):
        # VALIDATION LOGIC: One-sided wide data is allowed; mixed wide/long schemas are ambiguous.
        _column_names(self, ('bond', 'known_time', 'dealer'))
        wide, long = self.bid is not None or self.ask is not None, self.side is not None or self.value is not None
        if wide == long or (long and (self.side is None or self.value is None)):
            raise ValueError('Quotes require either bid/ask columns or both side and value columns.')
        if long and (self.bid_size is not None or self.ask_size is not None):
            raise ValueError('Long quotes use quantity; bid_size/ask_size require wide quotes.')


def _column_names(schema, required):
    # VALIDATION LOGIC: Check contracts without touching any source table.
    for name, value in asdict(schema).items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'{name} must be a nonempty column-name string or None.')
        if name in required and value is None:
            raise ValueError(f'{name} is a required column mapping.')


@dataclass(frozen=True)
class PipelineConfig:
    """Normalize target/anchor/cpp and quotes into compatible units before comparing them.

    target_scale multiplies target, actual, anchor, proxy (cpp), and rollover adjustment;
    quote_scale multiplies quote spreads. error_scale converts prediction errors into unit.
    quantity_scale applies to transaction sizes; quote_quantity_scale independently
    scales quote sizes into the same unit. State is always same-day.
    """
    # CONFIGURATION LOGIC: Defaults are declared constants, never selected using evaluation outcomes.
    transactions: TransactionColumns
    quotes: QuoteColumns
    value_kind: str = 'spread'
    error_scale: float = 1.0
    unit: str = None
    timezone: str = 'America/New_York'
    quote_timezone: str = None
    age_min: float = 30.0
    sync_min: float = 1.0
    lookback_min: float = 30.0
    clip_floor: float = 10.0
    allow_exact: bool = True
    case_seed: int = 2026
    quantity_scale: float = 1.0
    quote_quantity_scale: float = 1.0
    quote_scale: float = 1.0
    target_scale: float = 1.0
    selection_min_count: int = 30
    synthetic: bool = False
    priority_history_column: str = None
    priority_cpp_gap_column: str = None

    def __post_init__(self):
        # VALIDATION LOGIC: Reject ambiguous units and impossible feature settings before ingesting data.
        if not isinstance(self.transactions, TransactionColumns) or not isinstance(self.quotes, QuoteColumns):
            raise TypeError('Use TransactionColumns and QuoteColumns, or PipelineConfig.from_dict().')
        if self.value_kind != 'spread':
            raise ValueError("Only spread input is supported; convert prices upstream and use value_kind='spread'.")
        if self.unit is None:
            object.__setattr__(self, 'unit', 'bps')
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError('unit must be a nonempty displayed error-unit name.')
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError('timezone must be a named timezone.')
        if self.quote_timezone is not None and (not isinstance(self.quote_timezone, str) or not self.quote_timezone.strip()):
            raise ValueError('quote_timezone must be a named timezone or None.')
        if self.priority_history_column is not None and (not isinstance(self.priority_history_column, str) or not self.priority_history_column.strip()):
            raise ValueError('priority_history_column must be a nonempty column name or None.')
        if self.priority_cpp_gap_column is not None and (not isinstance(self.priority_cpp_gap_column, str) or not self.priority_cpp_gap_column.strip()):
            raise ValueError('priority_cpp_gap_column must be a nonempty column name or None.')
        for name in ['error_scale', 'quantity_scale', 'quote_quantity_scale', 'quote_scale', 'target_scale', 'clip_floor', 'age_min']:
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite.')
        for name in ['sync_min', 'lookback_min']:
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite.')
        if isinstance(self.selection_min_count, bool) or not isinstance(self.selection_min_count, int) or self.selection_min_count < 1:
            raise ValueError('selection_min_count must be a positive integer.')
        if not isinstance(self.allow_exact, bool) or not isinstance(self.synthetic, bool):
            raise ValueError('allow_exact and synthetic must be booleans.')
        if isinstance(self.case_seed, bool) or not isinstance(self.case_seed, int) or self.case_seed < 0:
            raise ValueError('case_seed must be a nonnegative integer.')

    def to_dict(self):
        # SERIALIZATION LOGIC: Dataclass mappings and scalar settings form a JSON-compatible CLI contract.
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        # CONFIGURATION LOGIC: Construct nested mappings explicitly; unknown fields raise instead of disappearing.
        values = dict(values)
        for name, schema in [('transactions', TransactionColumns), ('quotes', QuoteColumns)]:
            if name not in values:
                raise ValueError(f'Configuration needs {name!r}.')
            if not isinstance(values[name], schema):
                values[name] = schema(**values[name])
        return cls(**values)
