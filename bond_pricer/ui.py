"""An explicit-Apply notebook workbench for two-model prediction diagnostics."""
# SETUP LOGIC: Widgets and figures are created on demand; importing this module reads no data.
from html import escape
from io import BytesIO
import numpy as np
from pandas.api.types import is_numeric_dtype
import ipywidgets as w
from IPython.display import display
from .data import read_data
from .engine import Comparison, compare_predictions
from .slices import Slice, default_slices
from . import plots


def _edges(text):
    # CONFIGURATION LOGIC: Parse literal numerical boundaries only; no eval or executable expressions.
    if not text.strip():
        return None
    try:
        return [float(value.strip()) for value in text.split(',')]
    except ValueError as exc:
        raise ValueError('Bin edges must be comma-separated numbers, e.g. -inf,0,1000000,inf.') from exc


def _image(figure):
    # PLOTTING LOGIC: Render one complete PNG, avoiding the duplicate pyplot notebook display path.
    buffer = BytesIO()
    figure.savefig(buffer,format='png',dpi=135)
    return w.Image(value=buffer.getvalue(),format='png',layout=w.Layout(width='100%'))


def _preview(table):
    # UI LOGIC: Bound the on-screen table; complete tables remain in the export and public result API.
    columns = [c for c in ['group','x','y','n','reference_mae','candidate_mae','mae_improvement_pct','p95_delta','low_support'] if c in table]
    shown = table.loc[:,columns].head(100)
    note = f'<p>Showing {len(shown):,} of {len(table):,} groups. Export contains every group and metric.</p>'
    return w.HTML(note+'<div style="max-height:420px;overflow:auto">'+shown.to_html(index=False,escape=True,float_format=lambda x:f'{x:.5g}')+'</div>')


def _filter_text(condition):
    # UI LOGIC: Present the recorded population in readable terms instead of a Python configuration dict.
    column, parts = condition['column'], []
    if condition['minimum'] is not None:
        parts.append(f'{column} {">=" if condition["minimum_inclusive"] else ">"} {condition["minimum"]:,.8g}')
    if condition['maximum'] is not None:
        parts.append(f'{column} {"<=" if condition["maximum_inclusive"] else "<"} {condition["maximum"]:,.8g}')
    if condition['values'] is not None:
        parts.append(f'{column} in '+', '.join(str(v) for v in condition['values']))
    text = ' and '.join(parts) or f'{column} is present'
    return text+(' (including missing)' if condition['include_missing'] else '')


class ComparisonPanel:
    # UI LOGIC: Input edits do not alter the applied comparison or exported results until Apply succeeds.
    def __init__(self, data=None, *, actual=None, predictions=None, id_column=None, time_column=None,
                 bond_column=None, error_scale=1, unit='bps', timezone='America/New_York',
                 default_slices=None, tolerance=1, reference_anchor=None, candidate_anchor=None):
        self.result, self.applied_specs, self.busy = None, None, False
        self.supplied_slices = default_slices
        self.prediction_mapping = predictions
        self.anchors = dict(reference_anchor=reference_anchor,candidate_anchor=candidate_anchor)
        self.initial_filters = []
        self.initial_names = None
        if isinstance(data,Comparison):
            existing = data
            actual, id_column = existing.config['actual'], existing.config['id_column']
            time_column, bond_column = existing.config['time_column'], existing.config['bond_column']
            error_scale, unit, timezone = existing.config['error_scale'], existing.unit, existing.config['timezone']
            tolerance = existing.tolerance
            predictions = {existing.reference_name:existing.config['reference'],existing.candidate_name:existing.config['candidate']}
            self.prediction_mapping = predictions
            self.anchors = {k:existing.config[k] for k in ['reference_anchor','candidate_anchor']}
            self.initial_filters = list(existing.filter_history)
            self.initial_names = (existing.reference_name, existing.candidate_name)
            data = existing.data
        self.data = read_data(data, string_columns='all') if data is not None else None
        self.path = w.Text(description='Data path:',placeholder='Local CSV / Parquet path',layout=w.Layout(width='80%'))
        self.load = w.Button(description='Load data',button_style='info')
        self.load.on_click(self.load_data)
        self.actual = w.Dropdown(description='Actual:')
        self.reference, self.candidate = w.Dropdown(description='Reference:'),w.Dropdown(description='Candidate:')
        self.identity, self.time, self.bond = [w.Dropdown(description=label) for label in ['Trade ID:','Timestamp:','Bond ID:']]
        self.scale, self.unit, self.zone = w.FloatText(value=error_scale,description='Error scale:'),w.Text(value=unit,description='Unit:'),w.Text(value=timezone,description='Timezone:')
        self.tolerance = w.FloatText(value=tolerance,description='Tolerance:')
        self.first, self.second = w.Dropdown(description='Slice:'),w.Dropdown(description='Cross with:')
        self.first_bins, self.second_bins = w.Text(description='Slice bins:'),w.Text(description='Cross bins:')
        self.first_right, self.second_right = w.Checkbox(value=True,description='Slice bins (a,b]'),w.Checkbox(value=True,description='Cross bins (a,b]')
        self.minimum = w.BoundedIntText(value=30,min=1,max=100000000,description='Minimum N:')
        self.top_n = w.BoundedIntText(value=20,min=1,max=50,description='Top groups:')
        self.metric = w.Dropdown(options=[(label,key) for key,label in plots.METRICS.items()],description='Metric:',layout=w.Layout(width='470px'))
        self.filters = [self.make_filter(),self.make_filter()]
        self.anchor_mapping = {}
        self.apply = w.Button(description='Apply comparison',button_style='primary')
        self.apply.on_click(self.run)
        self.export_path = w.Text(value='outputs/model_comparisons',description='Export to:',layout=w.Layout(width='70%'))
        self.export_button = w.Button(description='Export applied review',disabled=True)
        self.export_button.on_click(self.export)
        self.status = w.HTML('Load a table or pass a DataFrame. This workbench never fits a model.')
        self.applied = w.HTML()
        self.views = w.Tab(children=[w.HTML('Apply to calculate the common sample.')])
        self.views.set_title(0,'Results')
        inputs = w.VBox([w.HBox([self.path,self.load]),w.HBox([self.actual,self.reference,self.candidate]),
                         w.HBox([self.identity,self.time,self.bond]),w.HBox([self.scale,self.unit]),w.HBox([self.zone,self.tolerance])])
        options = w.Accordion(children=[inputs])
        options.set_title(0,'Data and prediction columns')
        controls = w.VBox([w.HBox([self.first,self.second]),w.HBox([self.first_bins,self.first_right]),
                           w.HBox([self.second_bins,self.second_right]),w.HBox([self.minimum,self.top_n,self.metric])])
        self.widget = w.VBox([w.HTML('<h3>Bond pricer · Compare two models</h3><p>Choose a reference, candidate and slices. Negative error delta means improvement. Units must match before comparison.</p>'),
                              options,controls,w.HTML('<b>Optional population filters</b> · two rows form an intersection; blank = no restriction'),
                              *[item['widget'] for item in self.filters],self.apply,self.status,self.applied,self.views,
                              w.HBox([self.export_path,self.export_button])])
        self.first.observe(lambda _:self.set_bins(self.first,self.first_bins,self.first_right),names='value')
        self.second.observe(lambda _:self.set_bins(self.second,self.second_bins,self.second_right),names='value')
        if self.data is not None:
            self.configure(actual,id_column,time_column,bond_column)

    def make_filter(self):
        # UI LOGIC: Two optional filters cover common large-trade × maturity/sparsity reviews.
        fields = dict(column=w.Dropdown(description='Column:'),low=w.Text(description='Lower:'),high=w.Text(description='Upper:'),
                      strict=w.Checkbox(value=False,description='Strict lower >'),categories=w.Text(description='Values:',placeholder='A; B; C (optional)'))
        fields['widget'] = w.VBox([w.HBox([fields['column'],fields['low'],fields['high']]),w.HBox([fields['strict'],fields['categories']])])
        return fields

    def configure(self, actual=None, identity=None, time=None, bond=None):
        # UI LOGIC: Populate selectors without guessing hidden metadata or running a prediction.
        columns = list(self.data.columns)
        if not columns:
            raise ValueError('The input table has no columns.')
        self.actual.options = columns
        self.actual.value = actual if actual in columns else next((c for c in ['actual','BM_SPREAD','target'] if c in columns),columns[0])
        mapping = self.prediction_mapping or {str(c):c for c in columns if c != self.actual.value}
        if len(mapping) < 2:
            raise ValueError('Supply at least two prediction columns, or use compare_models first.')
        if not set(mapping.values()).issubset(columns):
            raise ValueError('A configured prediction column does not exist in this data.')
        self.mapping = mapping
        self.reference.options = self.candidate.options = list(mapping)
        preferred_reference = self.initial_names[0] if self.initial_names else None
        self.reference.value = preferred_reference or next((c for c in ['BASE','Base','base_prediction','pred_base','Reference prediction'] if c in mapping),list(mapping)[0])
        others = [c for c in mapping if c != self.reference.value]
        preferred_candidate = self.initial_names[1] if self.initial_names else None
        self.candidate.value = preferred_candidate or next((c for c in ['Candidate','candidate_prediction','new_prediction','pred_new','Candidate prediction'] if c in others),others[0])
        if not self.anchor_mapping:
            self.anchor_mapping = {self.reference.value:self.anchors['reference_anchor'],
                                   self.candidate.value:self.anchors['candidate_anchor']}
        for control,value,aliases in [(self.identity,identity,['row_id','trade_id']),(self.time,time,['time','EFFECTIVE_DATETIME_TS','timestamp']),
                                      (self.bond,bond,['CUSIP','cusip','bond_id'])]:
            control.options = [('Not supplied',None)]+[(str(c),c) for c in columns]
            control.value = value if value in columns else next((c for c in aliases if c in columns),None)
        supplied = self.supplied_slices
        defaults = supplied if supplied is not None else default_slices(columns)
        defaults = [spec for spec in defaults if self.time.value is not None or spec.column not in {'__hour', '__date'}]
        self.defaults = {s.column:s for s in defaults}
        slice_columns = list(dict.fromkeys(columns+(['__hour','__date'] if self.time.value else [])))
        self.first.options = [(str(c),c) for c in slice_columns]
        self.second.options = [('None',None)]+[(str(c),c) for c in slice_columns]
        predicted = set(mapping.values()) if self.prediction_mapping else {mapping[self.reference.value],mapping[self.candidate.value]}
        excluded = predicted | {self.actual.value,self.identity.value,self.time.value,self.bond.value}
        metadata = [c for c in columns if c not in excluded]
        categorical = [c for c in metadata if not is_numeric_dtype(self.data[c])]
        preferred = next(iter(categorical or metadata), columns[0])
        self.first.value = next(iter(self.defaults), preferred)
        self.second.value = None
        self.set_bins(self.first,self.first_bins,self.first_right)
        for item in self.filters:
            item['column'].options = [('No filter',None)]+[(str(c),c) for c in columns]
        self.status.value = f'Loaded {len(self.data):,} rows and {len(columns)} columns. Select predictions and Apply.'

    def set_bins(self, selector, text, right):
        # UI LOGIC: Use fixed domain defaults where available; new columns accept custom bin edges.
        spec = getattr(self,'defaults',{}).get(selector.value)
        text.value = ','.join(str(v) for v in spec.bins) if spec and spec.bins is not None else ''
        right.value = spec.right if spec else True

    def load_data(self, _=None):
        # FILE IO LOGIC: Replace the input only on an explicit load; preserve the applied result on error.
        try:
            data = read_data(self.path.value, string_columns='all')
            self.data = data
            self.initial_filters = []
            self.initial_names = None
            self.prediction_mapping,self.anchor_mapping = None,{}
            self.anchors = dict(reference_anchor=None,candidate_anchor=None)
            self.configure()
        except Exception as exc:
            self.status.value = '<b>Input error:</b> '+escape(str(exc))

    def specs(self):
        # CONFIGURATION LOGIC: Snapshot the currently requested boundaries, names and closure.
        first = self.make_spec(self.first,self.first_bins,self.first_right,self.top_n.value)
        second = self.make_spec(self.second,self.second_bins,self.second_right,min(self.top_n.value,20)) if self.second.value else None
        return first,second

    def make_spec(self, selector, text, right, top_n):
        # CONFIGURATION LOGIC: Friendly default labels remain valid only while their boundaries and closure match.
        bins = _edges(text.value)
        original = self.defaults.get(selector.value)
        same_bins = original is not None and ((bins is None and original.bins is None) or
                    (bins is not None and original.bins is not None and np.array_equal(bins,original.bins)))
        labels = original.labels if same_bins and right.value == original.right else None
        return Slice(selector.value,bins,labels,right.value,top_n)

    def run(self, _=None):
        # UI LOGIC: Publish result state only after a successful computation; previous exports remain valid.
        if self.busy:
            return
        self.busy, self.apply.disabled = True, True
        self.status.value = 'Comparing saved predictions on common rows...'
        try:
            if self.data is None:
                raise ValueError('Load data first.')
            reference,candidate = self.reference.value,self.candidate.value
            result = compare_predictions(self.data,self.actual.value,self.mapping[reference],self.mapping[candidate],
                                         reference_name=reference,candidate_name=candidate,id_column=self.identity.value,
                                         time_column=self.time.value,bond_column=self.bond.value,error_scale=self.scale.value,
                                         unit=self.unit.value,timezone=self.zone.value,tolerance=self.tolerance.value,
                                         reference_anchor=self.anchor_mapping.get(reference),candidate_anchor=self.anchor_mapping.get(candidate))
            result.filter_history = list(self.initial_filters)
            filters = []
            for item in self.filters:
                has_condition = any(item[k].value.strip() for k in ['low','high','categories'])
                if item['column'].value is not None and has_condition:
                    options = dict(minimum=float(item['low'].value) if item['low'].value.strip() else None,
                                   maximum=float(item['high'].value) if item['high'].value.strip() else None,
                                   values=[v.strip() for v in item['categories'].value.split(';')] if item['categories'].value.strip() else None,
                                   minimum_inclusive=not item['strict'].value)
                    result = result.filter(item['column'].value,**options)
                    filters.append(dict(column=item['column'].value,**options))
            if result.rows.empty:
                raise ValueError(f'No common finite rows in this population. Coverage: {result.coverage}')
            first,second = self.specs()
            table = result.cross_slice(first,second,min_count=self.minimum.value) if second else result.slice(first,min_count=self.minimum.value)
            title = f'{candidate} vs {reference} | {first.column}'+(f' × {second.column}' if second else '')
            chart = plots.heatmap(table,self.metric.value,unit=result.unit,title=title) if second else plots.slice_figure(table,self.metric.value,unit=result.unit,title=title)
            self.views.children = [_image(plots.overview(result)),w.VBox([_image(chart),_preview(table)]),
                                   w.VBox([_image(plots.daily_figure(result.daily(),unit=result.unit)),w.HTML(result.stability().to_html(index=False,escape=True))]),
                                   w.HTML(result.missingness().to_html(index=False,escape=True))]
            for i,label in enumerate(['Overview','Slices','Dates','Missingness']):
                self.views.set_title(i,label)
            self.result,self.applied_specs = result,(first,second)
            self.applied_filters,self.applied_min_count = result.filter_history,self.minimum.value
            self.applied_metric = self.metric.value
            self.export_button.disabled = False
            population = '; '.join(_filter_text(f) for f in result.filter_history) or 'All supplied trades'
            self.applied.value = '<b>Applied:</b> '+escape(title)+f' | {len(result.rows):,} paired trades<br><b>Population:</b> '+escape(population)
            self.status.value = 'Ready. Controls affect the next Apply. Export always uses the applied result. Low N is flagged; date sensitivity is descriptive.'
        except Exception as exc:
            self.status.value = '<b>Comparison error:</b> '+escape(str(exc))+' Previous applied result is unchanged.'
        finally:
            self.busy,self.apply.disabled = False,False

    def export(self, _=None):
        # FILE IO LOGIC: Export the applied data/specifications, never newly edited but unapplied controls.
        if self.result is None or self.busy:
            return
        try:
            first,second = self.applied_specs
            output = self.result.export(self.export_path.value,slices=[first],interactions=[(first,second)] if second else [],
                                        min_count=self.applied_min_count,metric=self.applied_metric)
            self.status.value = 'Saved complete PNG, CSV and HTML review to '+escape(str(output))
        except Exception as exc:
            self.status.value = '<b>Export error:</b> '+escape(str(exc))

    def show(self):
        # UI LOGIC: Display one workbench instance; the user explicitly applies a comparison.
        display(self.widget)
        return self


def show_comparison(data=None, **kwargs):
    # UI LOGIC: One public notebook entry for DataFrames, paths or an existing Comparison.
    return ComparisonPanel(data,**kwargs).show()
