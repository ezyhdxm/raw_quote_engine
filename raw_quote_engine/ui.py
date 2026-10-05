"""A notebook form for explicitly mapped, caller-prepared spread data."""
# SETUP LOGIC: Creating a form never normalizes a dataset, builds quote features, or fits a model.
from dataclasses import asdict
from html import escape
import json
from zoneinfo import ZoneInfo
import pandas as pd
import ipywidgets as w
from IPython.display import display
from .config import PipelineConfig, QuoteColumns, TransactionColumns
from .training import TrainingConfig
from .pipeline import run_research


def _frame_columns(frame, label):
    # VALIDATION LOGIC: Dropdowns represent actual named source columns, never inferred aliases.
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{label} must be a pandas DataFrame loaded and prepared by the caller.')
    if frame.columns.duplicated().any():
        raise ValueError(f'{label} column names must be unique.')
    if any(not isinstance(column, str) or not column.strip() for column in frame.columns):
        raise ValueError(f'{label} column names must be nonempty strings.')
    return list(frame.columns)


def _feature_list(values, columns, label):
    # VALIDATION LOGIC: Catch accidental string arguments, duplicates, and missing source features early.
    if isinstance(values, str):
        raise TypeError(f'{label} must be a sequence of column names, not one string.')
    result = list(values)
    if len(set(result)) != len(result) or set(result) - set(columns):
        raise ValueError(f'{label} needs unique column names present in the transaction DataFrame.')
    return result


class ResearchForm:
    """Editable configuration, structural preflight, and an explicit validation-run button.

    ``validate()`` raises on invalid mappings but does not scan all records or train.
    ``start()`` performs full engine checks and returns a ResearchRun. Errors also appear
    inline, and controls are restored even when validation or fitting fails.
    """
    # UI LOGIC: Retain references to supplied frames; the engine owns any work copies when Start is pressed.
    def __init__(self, transactions_df, quotes_df, base_features, base_cat_features, model_params,
                 *, config=None, training=None, output='runs/research', cache_dir=None, quote_universe=None):
        self.transactions, self.quotes = transactions_df, quotes_df
        self.quote_universe = quote_universe
        self.tx_columns = _frame_columns(transactions_df, 'transactions_df')
        self.quote_columns = _frame_columns(quotes_df, 'quotes_df')
        base = _feature_list(base_features, self.tx_columns, 'BASE_FEATURES')
        categorical = _feature_list(base_cat_features, base, 'BASE_CAT_FEATURES')
        self._config = config.to_dict() if isinstance(config, PipelineConfig) else dict(config or {})
        if self._config.get('value_kind', 'spread') != 'spread':
            raise ValueError('This engine accepts spread data only; convert prices upstream before configuring a run.')
        default_training = TrainingConfig(selection_slice='all', apply_model_defaults=False)
        self._training = default_training.to_dict()
        self._training.update(training.to_dict() if isinstance(training, TrainingConfig) else dict(training or {}))
        self.controls, self.run, self.running = {}, None, False
        self.status, self.progress = w.HTML(), w.FloatProgress(min=0, max=1, value=0)
        self.review_output = w.Output()
        self._make_mappings()
        self._make_features(base, categorical, model_params)
        self._make_settings(output, cache_dir)
        self._make_widget()

    def _add(self, key, control):
        # UI LOGIC: Public named controls support notebook customization and simple programmatic checks.
        control.style.description_width = '190px'
        control.layout.width = 'min(100%, 650px)'
        self.controls[key] = control
        return control

    def _column(self, key, label, columns, initial=None, required=False):
        # UI LOGIC: Mandatory mappings remain blank until supplied; optional fields explicitly allow None.
        placeholder = '(Select a required column)' if required else '(None)'
        if initial is not None and initial not in columns:
            raise ValueError(f'{label}: configured column {initial!r} is absent from the supplied DataFrame.')
        return self._add(key, w.Dropdown(options=[(placeholder, None)]+[(c, c) for c in columns],
                                        value=initial, description=label+':'))

    def _make_mappings(self):
        # UI LOGIC: Source-column selection has no dataset-specific names or feature recombination.
        tx = self._config.get('transactions', {})
        quote = self._config.get('quotes', {})
        required = {'bond': 'Bond identifier', 'time': 'Prediction time', 'target': 'Training target'}
        optional = {'id': 'Unique record ID', 'actual': 'Actual spread level', 'anchor': 'Prediction-time anchor',
                    'rollover_adjustment': 'Rollover adjustment', 'cpp': 'Reference proxy spread (optional)',
                    'sector': 'Sector', 'issuer': 'Issuer', 'quantity': 'Transaction quantity',
                    'prev_quantity': 'Previous quantity', 'maturity_date': 'Maturity date',
                    'maturity_years': 'Remaining maturity years', 'split': 'Explicit split labels'}
        for name, label in {**required, **optional}.items():
            self._column('tx_'+name, label, self.tx_columns, tx.get(name), name in required)
        self._add('quote_format', w.Dropdown(options=['wide', 'long'],
                  value='long' if quote.get('side') or quote.get('value') else 'wide', description='Quote layout:'))
        quote_fields = {'bond': 'Bond identifier', 'known_time': 'Known / received time', 'dealer': 'Dealer identifier',
                        'bid': 'Bid spread', 'ask': 'Ask spread', 'side': 'Side (bid / ask)', 'value': 'Spread value',
                        'quantity': 'Common / long size', 'bid_size': 'Bid size', 'ask_size': 'Ask size',
                        'original_timestamp': 'Original timestamp (audit)', 'source_id': 'Source record ID'}
        for name, label in quote_fields.items():
            self._column('quote_'+name, label, self.quote_columns, quote.get(name),
                         name in {'bond', 'known_time', 'dealer'})
        self.controls['quote_format'].observe(self._layout_changed, names='value')
        self._layout_changed()

    def _layout_changed(self, change=None):
        # UI LOGIC: Preserve hidden selections for convenience, but configuration excludes inactive mappings.
        long = self.controls['quote_format'].value == 'long'
        for name in ['bid', 'ask', 'bid_size', 'ask_size']:
            self.controls['quote_'+name].layout.display = 'none' if long else ''
        for name in ['side', 'value']:
            self.controls['quote_'+name].layout.display = '' if long else 'none'

    def _make_features(self, base, categorical, params):
        # UI LOGIC: Existing feature order is retained; choices only reference supplied transaction columns.
        options = base+[column for column in self.tx_columns if column not in base]
        self._add('base_features', w.SelectMultiple(options=options, value=tuple(base), rows=9,
                                                   description='BASE_FEATURES:'))
        self._add('base_cat_features', w.SelectMultiple(options=base, value=tuple(categorical), rows=5,
                                                       description='BASE_CAT_FEATURES:'))
        self._add('model_params', w.Textarea(value=json.dumps(params, indent=2, allow_nan=False),
                                             description='LGB_PARAMS (JSON):', layout=w.Layout(height='200px')))
        self.controls['base_features'].observe(self._features_changed, names='value')

    def _features_changed(self, change):
        # UI LOGIC: Dropping a base feature also drops its categorical designation, never adds a feature.
        control = self.controls['base_cat_features']
        retained = tuple(value for value in control.value if value in change['new'])
        control.options, control.value = list(change['new']), retained

    def _make_settings(self, output, cache_dir):
        # CONFIGURATION LOGIC: Unit/time conversion is declared explicitly; the form supports spreads only.
        numeric = {'target_scale': ('Transaction spread scale', 1.), 'quote_scale': ('Quote spread scale', 1.),
                   'error_scale': ('Displayed error scale', 1.), 'quantity_scale': ('Transaction size scale', 1.),
                   'quote_quantity_scale': ('Quote size scale', 1.), 'age_min': ('Quote freshness (minutes)', 30.),
                   'sync_min': ('Side synchronization (minutes)', 1.), 'lookback_min': ('Movement window (minutes)', 30.),
                   'clip_floor': ('Robust clipping floor', 10.)}
        for name, (label, default) in numeric.items():
            self._add(name, w.FloatText(value=self._config.get(name, default), description=label+':'))
        for name, label, default in [('timezone', 'Transaction / report timezone', 'America/New_York'),
                                     ('quote_timezone', 'Naive quote timezone', ''), ('unit', 'Displayed spread unit', 'bps')]:
            self._add(name, w.Text(value=self._config.get(name) or default, description=label+':'))
        self._add('allow_exact', w.Checkbox(value=self._config.get('allow_exact', True),
                                           description='Include quotes known exactly at prediction time'))
        self._column('priority_history_column', 'Prior trade count (optional)', self.tx_columns,
                     self._config.get('priority_history_column'))
        self._column('priority_cpp_gap_column', 'Precomputed proxy gap (bps)', self.tx_columns,
                     self._config.get('priority_cpp_gap_column'))
        self._add('target_mode', w.Dropdown(options=['level', 'delta'], value=self._training['target_mode'],
                                            description='Training target mode:'))
        self._add('selection_slice', w.Dropdown(options=[('All records', 'all'), ('Large trades', 'large'),
                                                        ('Large trades; maturity >1y', 'large_long')],
                   value=self._training['selection_slice'], description='Validation selection slice:'))
        self._make_split_controls()
        self._add('category_order', w.Dropdown(options=['sorted', 'appearance'],
                   value=self._training['category_order'], description='Training category order:'))
        self._add('apply_model_defaults', w.Checkbox(value=self._training['apply_model_defaults'],
                    description='Add engine model defaults (off = use supplied LGB_PARAMS)'))
        self._add('output', w.Text(value=str(output), description='Run output directory:'))
        self._add('cache_dir', w.Text(value=str(cache_dir) if cache_dir is not None else '', description='Shared cache (optional):'))

    def _make_split_controls(self):
        # UI LOGIC: An explicit split mapping takes precedence; otherwise observed local dates define blocks.
        for name, label in [('validation_fraction', 'Validation date fraction'), ('test_fraction', 'Test date fraction'),
                            ('useful_improvement_pct', 'Useful improvement (%)')]:
            self._add(name, w.FloatText(value=self._training[name], description=label+':'))
        for name, label in [('embargo_dates', 'Embargo before validation'), ('min_train_dates', 'Minimum training dates'),
                            ('min_selection_rows', 'Minimum selection records')]:
            self._add(name, w.IntText(value=self._training[name], description=label+':'))
        for name, label in [('validation_dates', 'Validation dates (override)'), ('test_dates', 'Test dates (override)'),
                            ('test_embargo_dates', 'Embargo before test (override)')]:
            initial = self._training.get(name)
            self._add(name, w.Text(value='' if initial is None else str(initial), description=label+':',
                                  placeholder='Blank = use fraction / shared embargo'))

    def _section(self, keys, note=''):
        # UI LOGIC: Sections are compact and expandable; long source tables are never printed.
        children = [w.HTML(note)] if note else []
        return w.VBox(children+[self.controls[name] for name in keys])

    def _make_widget(self):
        # UI LOGIC: Explain caller responsibilities before exposing a potentially expensive run action.
        intro = w.HTML('<h3>Raw quote research — spread data</h3><p>Supply prepared transaction and quote DataFrames, '
            'BASE_FEATURES, BASE_CAT_FEATURES and LGB_PARAMS. Map columns below; no columns are guessed.</p>'
            '<p><b>Your preparation:</b> load/filter the intended transaction universe and period; construct causal base '
            'features; resolve identifiers, sector/issuer metadata, timestamps and benchmark-consistent spreads; '
            'prepare any proxy, anchor and signed rollover adjustment. The engine does not build a proxy or combine '
            'categorical features for you. Provide both categorical columns directly as base features when needed. '
            'Sizes, spread units and timezones must be declared correctly. Prices are unsupported.</p>')
        sections = [self._section([k for k in self.controls if k.startswith('tx_')],
                    'Required: bond, prediction time, target. Optional fields may stay (None).'),
            self._section(['quote_format']+[k for k in self.controls if k.startswith('quote_') and k not in
                          {'quote_format', 'quote_scale', 'quote_quantity_scale', 'quote_timezone'}],
                    'Wide: at least one bid/ask spread. Long: side and spread value; sides bid/b/ask/a/offer. '
                    'Known time is when the quote was available; original time is audit-only.'),
            self._section(['base_features', 'base_cat_features', 'model_params', 'category_order', 'apply_model_defaults']),
            self._section(['timezone', 'quote_timezone', 'target_scale', 'quote_scale', 'error_scale', 'unit',
                           'quantity_scale', 'quote_quantity_scale', 'age_min', 'sync_min', 'lookback_min',
                           'clip_floor', 'allow_exact'],
                    'Blank quote timezone = transaction timezone for naive quotes. Aware timestamps retain their '
                    'instant and convert to report timezone. All spread fields must share benchmark meaning; '
                    'multipliers only convert units. Transaction spread scale applies equally to target, actual, '
                    'anchor, proxy and rollover adjustment. Clipping floor uses scaled spread units. '
                    'Quantity scales must produce compatible sizes.'),
            self._section(['target_mode', 'selection_slice', 'validation_fraction', 'test_fraction', 'embargo_dates',
                           'validation_dates', 'test_dates', 'test_embargo_dates', 'min_train_dates',
                           'min_selection_rows', 'useful_improvement_pct', 'priority_history_column', 'priority_cpp_gap_column'],
                    'Level: prediction is the spread. Delta: spread = prediction + anchor − rollover adjustment '
                    '(zero when unmapped). Actual is an independent observed level; if absent, delta truth uses '
                    'target + anchor − adjustment. Explicit Train/Validation/Test/Embargo labels override date '
                    'allocation. This form never evaluates the reserved final test.'),
            self._section(['output', 'cache_dir'], 'Use a new run directory for changed settings. Completed matching stages reuse their cache.')]
        accordion = w.Accordion(children=sections, selected_index=0)
        for index, title in enumerate(['1. Transactions', '2. Raw quotes', '3. Baseline model', '4. Units and quote settings',
                                       '5. Chronological validation', '6. Save and resume']):
            accordion.set_title(index, title)
        sections[4].children += (w.HTML('Optional prior-count mapping must already be known at prediction time; '
            'the sparse-history slice uses counts 0–14. If omitted, diagnostics use engine-computed trailing '
            '30-day counts with history-coverage flags. An optional precomputed proxy gap is already in bps; '
            'its absolute value overrides proxy-minus-adjusted-anchor computation and is not rescaled.'),)
        self.validate_button = w.Button(description='Validate inputs', button_style='info')
        self.start_button = w.Button(description='Run validation', button_style='primary')
        self.review_button = w.Button(description='Open saved review', disabled=True)
        self.validate_button.on_click(self._validate_clicked)
        self.start_button.on_click(self._start_clicked)
        self.review_button.on_click(self._review_clicked)
        self.status.value = 'Choose mappings, then Validate inputs. Validation here checks structure only; it does not train.'
        self.widget = w.VBox([intro, accordion, w.HBox([self.validate_button, self.start_button, self.review_button]),
                              self.status, self.progress, self.review_output])

    def configuration(self):
        # CONFIGURATION LOGIC: Keep explicit advanced settings; replace only fields editable in this form.
        values = dict(self._config)
        tx = {name.removeprefix('tx_'): control.value for name, control in self.controls.items() if name.startswith('tx_')}
        quote_names = list(QuoteColumns.__dataclass_fields__)
        quote = {name: self.controls['quote_'+name].value for name in quote_names}
        inactive = ['bid', 'ask', 'bid_size', 'ask_size'] if self.controls['quote_format'].value == 'long' else ['side', 'value']
        for name in inactive:
            quote[name] = None
        values.update(transactions=TransactionColumns(**tx), quotes=QuoteColumns(**quote), value_kind='spread')
        for name in ['timezone', 'unit', 'target_scale', 'quote_scale', 'error_scale', 'quantity_scale',
                     'quote_quantity_scale', 'age_min', 'sync_min', 'lookback_min', 'clip_floor', 'allow_exact',
                     'priority_history_column', 'priority_cpp_gap_column']:
            values[name] = self.controls[name].value
        values['quote_timezone'] = self.controls['quote_timezone'].value.strip() or None
        return PipelineConfig(**values)

    def training_configuration(self):
        # CONFIGURATION LOGIC: Parse only literal integer overrides; no expressions or inference from data length.
        values = dict(self._training)
        for name in ['target_mode', 'selection_slice', 'validation_fraction', 'test_fraction', 'embargo_dates',
                     'min_train_dates', 'min_selection_rows', 'useful_improvement_pct', 'category_order', 'apply_model_defaults']:
            values[name] = self.controls[name].value
        for name in ['validation_dates', 'test_dates', 'test_embargo_dates']:
            text = self.controls[name].value.strip()
            try:
                values[name] = int(text) if text else None
            except ValueError as exc:
                raise ValueError(f'{name} must be a whole number or blank.') from exc
        return TrainingConfig(**values)

    def _model_parameters(self):
        # CONFIGURATION LOGIC: Model parameters are plain JSON and are never evaluated as Python code.
        try:
            result = json.loads(self.controls['model_params'].value)
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError('LGB_PARAMS must be a JSON object with finite numbers: '+str(exc)) from exc
        if not isinstance(result, dict):
            raise ValueError('LGB_PARAMS must be a JSON object, such as {"n_estimators": 200}.')
        return result

    def validate(self):
        """Check configuration and schema only; return True or raise an actionable error."""
        # VALIDATION LOGIC: No normalization, feature aggregation, model fitting, or full-table statistics occur here.
        tx_columns = _frame_columns(self.transactions, 'transactions_df')
        quote_columns = _frame_columns(self.quotes, 'quotes_df')
        if self.transactions.empty:
            raise ValueError('The transaction DataFrame is empty.')
        config, training = self.configuration(), self.training_configuration()
        for schema, columns in [(config.transactions, tx_columns), (config.quotes, quote_columns)]:
            missing = set(value for value in asdict(schema).values() if value is not None)-set(columns)
            if missing:
                raise ValueError(f'Mapped columns are absent from the current DataFrame: {sorted(missing)}')
        for name in ['priority_history_column', 'priority_cpp_gap_column']:
            selected = getattr(config, name)
            if selected is not None and selected not in tx_columns:
                raise ValueError(f'{name}: {selected!r} is absent from the current transaction DataFrame.')
        base = _feature_list(self.controls['base_features'].value, tx_columns, 'BASE_FEATURES')
        _feature_list(self.controls['base_cat_features'].value, base, 'BASE_CAT_FEATURES')
        if not base:
            raise ValueError('Select at least one BASE_FEATURES column.')
        forbidden = {config.transactions.target, config.transactions.actual, config.transactions.id,
                     config.transactions.time, config.transactions.split}
        if set(base) & forbidden:
            raise ValueError('Target, actual outcome, record ID, timestamp and split columns cannot be base features.')
        self._validate_contracts(config, training)
        self._model_parameters()
        if not self.controls['output'].value.strip():
            raise ValueError('Choose an output directory.')
        self._show_mapping(config)
        return True

    def _validate_contracts(self, config, training):
        # VALIDATION LOGIC: Catch obvious role mistakes without asserting unverified feature causality or data quality.
        for label, columns in [('transaction bond/time/target', [config.transactions.bond, config.transactions.time, config.transactions.target]),
                               ('quote bond/time/dealer', [config.quotes.bond, config.quotes.known_time, config.quotes.dealer])]:
            if len(set(columns)) != len(columns):
                raise ValueError(f'Use distinct columns for {label}.')
        if config.quotes.side is not None and config.quotes.side == config.quotes.value:
            raise ValueError('Long quote side and spread value must use different columns.')
        if config.quotes.bid is not None and config.quotes.bid == config.quotes.ask:
            raise ValueError('Bid and ask must use different columns; map just one for one-sided quotes.')
        if training.target_mode == 'delta' and config.transactions.anchor is None:
            raise ValueError('Delta targets require a prediction-time anchor mapping.')
        if training.target_mode != 'delta' and config.transactions.rollover_adjustment is not None:
            raise ValueError('Rollover adjustment is applied only to delta targets; unmap it for level targets.')
        if training.selection_slice in {'large', 'large_long'} and config.transactions.quantity is None:
            raise ValueError('The selected large-trade slice needs a transaction quantity mapping.')
        if training.selection_slice == 'large_long' and not (config.transactions.maturity_date or config.transactions.maturity_years):
            raise ValueError('The selected maturity slice needs a maturity mapping.')
        for zone in [config.timezone, config.quote_timezone]:
            if zone:
                try:
                    ZoneInfo(zone)
                except Exception as exc:
                    raise ValueError(f'Unknown timezone {zone!r}; use an IANA timezone such as UTC or America/New_York.') from exc

    def _show_mapping(self, config):
        # UI LOGIC: Display at most three raw values per key field; never present this as a full data audit.
        fields = [('Transaction target', self.transactions, config.transactions.target),
                  ('Transaction time', self.transactions, config.transactions.time),
                  ('Quote known time', self.quotes, config.quotes.known_time)]
        preview = '<br>'.join(escape(label+': '+column+' → '+repr(frame[column].head(3).tolist()))
                              for label, frame, column in fields)
        self.status.value = '<b>Structural checks passed.</b> '+f'{len(self.transactions):,} transaction rows; {len(self.quotes):,} raw quote rows. '
        self.status.value += 'Counts are input sizes, not verified eligible samples.<br>'+preview
        self.status.value += '<br>Full key/time/numeric checks, chronological splits, quote support and model results are checked during Run validation.'

    def _progress(self, stage, current, total, message):
        # UI LOGIC: Progress is stage-local; do not imply a measured overall ETA from unknown aggregation work.
        self.status.value = '<b>'+escape(str(stage))+'</b>: '+escape(str(message))
        self.progress.value = min(1., max(0., current/total)) if current is not None and total else 0.
        self.progress.description = str(stage)[:18]

    def start(self):
        """Run validation explicitly; retain the returned run for review and later deliberate final-test use."""
        # ORCHESTRATION LOGIC: The same five supplied objects enter the generic engine; no preset or preparation step is hidden.
        if self.running:
            raise RuntimeError('A research run is already active in this form.')
        self.running = True
        self.start_button.disabled = self.validate_button.disabled = self.review_button.disabled = True
        try:
            self.validate()
            config, training = self.configuration(), self.training_configuration()
            self.run = run_research(self.transactions, self.quotes, list(self.controls['base_features'].value),
                self._model_parameters(), config, base_cat_features=list(self.controls['base_cat_features'].value),
                training=training, output=self.controls['output'].value.strip(),
                cache_dir=self.controls['cache_dir'].value.strip() or None, progress=self._progress,
                quote_universe=self.quote_universe, evaluate_test=False)
            self.status.value = '<b>Validation complete.</b> Report: '+escape(str(self.run.report))+'. Final test has not been requested by this form.'
            return self.run
        except Exception as exc:
            self._show_error(exc)
            raise
        finally:
            self.running = False
            self.start_button.disabled = self.validate_button.disabled = False
            self.review_button.disabled = self.run is None

    def _show_error(self, exc):
        # UI LOGIC: Escape source names and exception text rather than interpreting dataset content as HTML.
        self.status.value = '<b>Action required:</b> '+escape(str(exc))

    def _validate_clicked(self, button):
        # UI LOGIC: Notebook callbacks show concise inline errors; direct validate() still raises for callers/tests.
        try:
            self.validate()
        except Exception as exc:
            self._show_error(exc)

    def _start_clicked(self, button):
        # UI LOGIC: start() already records its failure and restores buttons; avoid a second traceback display.
        try:
            self.start()
        except Exception:
            pass

    def _review_clicked(self, button):
        # UI LOGIC: Review uses the saved completed run even when pending form edits differ; no models are refitted.
        if self.run is None:
            return
        with self.review_output:
            self.review_output.clear_output(wait=True)
            self.run.review()


def research_form(transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
                  *, config=None, training=None, output='runs/research', cache_dir=None, quote_universe=None):
    """Display once and return an editable ResearchForm; running remains a separate explicit action."""
    # UI LOGIC: Caller DataFrames are not printed, transformed, or fitted when the form is displayed.
    form = ResearchForm(transactions_df, quotes_df, BASE_FEATURES, BASE_CAT_FEATURES, LGB_PARAMS,
                        config=config, training=training, output=output, cache_dir=cache_dir,
                        quote_universe=quote_universe)
    display(form.widget)
    return form
