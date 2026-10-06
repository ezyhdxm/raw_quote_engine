"""Local reproducible HTML/CSV/PNG review bundles generated directly from comparison results."""
# SETUP LOGIC: Report rendering escapes user-supplied labels and never trains a model.
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import numpy as np
import pandas as pd
from .slices import Slice
from . import plots
from .ui_style import REPORT_STYLE

# UI LOGIC: Shared report selectors stay inside the report body; wide figures retain a readable display scale.
STYLE = REPORT_STYLE + '''
.analysis-report .plot-scroll{overflow-x:auto;max-width:100%;border:1px solid #e0e8ef;border-radius:10px;margin:16px 0;background:#fff}
.analysis-report .plot-scroll img{width:100%;max-width:none;margin:0;height:auto}
.analysis-report .figure-link{font-size:12px;margin-top:5px;color:#52697f}
.analysis-report .note{padding:16px 18px;background:#edf6f8;border-left:4px solid #217d88;border-radius:7px;color:#29485d}
.analysis-report pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6fa;border:1px solid #dce5ee;border-radius:8px;padding:16px;font-size:12px}
.analysis-report section{scroll-margin-top:18px}
@media print{.analysis-report .plot-scroll{overflow:visible}.analysis-report .plot-scroll img{min-width:0!important;max-width:100%}}
'''


def _table(frame):
    # FORMATTING LOGIC: Exact machine-readable values are exported separately; HTML is rounded for reading.
    return '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable results table">'+frame.to_html(
        index=False,escape=True,border=0,float_format=lambda x:f'{x:,.5g}')+'</div>'


def _figure(figure, output, filename, alt):
    # PLOTTING LOGIC: Keep the complete PNG and a readable minimum width for dense or narrow reviews.
    figure.savefig(output/filename,dpi=160)
    minimum = max(760,round(figure.get_figwidth()*80))
    return (f'<div class="plot-scroll" tabindex="0" role="region" aria-label="Scrollable figure"><img src="{filename}" '
            f'alt="{escape(alt)}" style="min-width:{minimum}px"></div>'
            f'<p class="figure-link"><a href="{filename}">Open full-size figure</a> · Scroll horizontally when the figure exceeds this pane.</p>')


def _json(value):
    # SERIALIZATION LOGIC: Convert configuration values into portable JSON without arbitrary object repr.
    if isinstance(value, Slice):
        return value.to_dict()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _portable(value):
    # SERIALIZATION LOGIC: Represent open bin endpoints as strings, keeping the manifest valid standard JSON.
    if isinstance(value,(Slice,np.generic,np.ndarray,Path)):
        return _portable(_json(value))
    if isinstance(value,dict):
        return {str(k):_portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_portable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None if np.isnan(value) else ('inf' if value > 0 else '-inf')
    return value


def export_comparison(comparison, folder, slices=None, interactions=None, min_count=30, metric='mae_delta'):
    # FILE IO LOGIC: Every click creates a separate self-contained review bundle.
    output = Path(folder)/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    specs = comparison.default_slices if slices is None else [s if isinstance(s,Slice) else Slice(s) for s in slices]
    interactions = interactions or []
    tables = {'summary':comparison.summary(min_count), 'daily':comparison.daily(min_count),
              'date_sensitivity':comparison.stability(), 'missingness':comparison.missingness()}
    # PLOTTING LOGIC: Overview and complete date history are exported at full resolution.
    overview = _figure(plots.overview(comparison),output,'overview.png','Common-sample metrics and coverage')
    daily = _figure(plots.daily_figure(tables['daily'],unit=comparison.unit),output,'daily.png','Daily losses')
    sections = ['<section id="overview"><h2>Common-sample overview</h2>'+_table(tables['summary'])+overview+'</section>']
    navigation = [('coverage','Coverage'),('overview','Overview')]
    # REPORTING LOGIC: Build every requested slice from the same result object used in the notebook.
    for i,spec in enumerate(specs):
        key = f'slice_{i+1:02d}'
        table = comparison.slice(spec,min_count=min_count)
        tables[key] = table
        title = spec.name or spec.column
        figure = _figure(plots.slice_figure(table,metric,unit=comparison.unit,title=title),output,f'{key}.png','Slice loss and support')
        sections.append(f'<section id="{key}"><h2>{escape(title)}</h2>'+figure+'<details><summary>Complete table</summary>'+_table(table)+'</details></section>')
        navigation.append((key,title))
    # REPORTING LOGIC: Cross tables preserve every observed cell and its support.
    for i,pair in enumerate(interactions):
        sx,sy = [s if isinstance(s,Slice) else Slice(s) for s in pair]
        key = f'interaction_{i+1:02d}'
        table = comparison.cross_slice(sx,sy,min_count=min_count)
        tables[key] = table
        if not table.empty:
            label = f'{sx.column} × {sy.column}'
            figure = _figure(plots.heatmap(table,metric,unit=comparison.unit,title=label),output,f'{key}.png','Loss and count heatmaps')
            sections.append(f'<section id="{key}"><h2>{escape(label)}</h2>'+figure+_table(table)+'</section>')
            navigation.append((key,label))
    # FILE IO LOGIC: Lossless CSV tables and the explicit review configuration accompany the figures.
    for name,table in tables.items():
        table.to_csv(output/f'{name}.csv',index=False)
    hashed = pd.util.hash_pandas_object(comparison.data,index=True,categorize=True).to_numpy().tobytes()
    schema = repr([(str(c),str(t)) for c,t in comparison.data.dtypes.items()]).encode()
    digest = hashlib.sha256(schema+hashed).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),config=comparison.config,
                    coverage=comparison.coverage,data_sha256=digest,slices=specs,
                    interactions=interactions,min_count=min_count,metric=metric,training_performed=False)
    manifest['filters'] = comparison.filter_history
    (output/'review.json').write_text(json.dumps(_portable(manifest),allow_nan=False,indent=2),encoding='utf-8')
    # REPORTING LOGIC: State denominators, signs and descriptive limitations next to the generated evidence.
    title = f'{comparison.candidate_name} vs {comparison.reference_name}'
    intro = f'<header class="hero"><div class="eyebrow">Saved prediction review</div><h1>{escape(title)}</h1>'
    intro += f'<p>Errors in {escape(comparison.unit)}. Record-weighted metrics on common finite targets and predictions.</p></header>'
    navigation.append(('dates','Dates and sensitivity'))
    intro += '<nav aria-label="Report sections">'+''.join(f'<a href="#{key}">{escape(label)}</a>' for key,label in navigation)+'</nav>'
    intro += '<p class="note">Negative MAE/P95 delta means improvement; positive MAE improvement % means improvement. '
    intro += 'Sparse cells are flagged, never removed. Slices are descriptive, overlap, and must not be added together. '
    intro += 'Date sensitivity is not a confidence interval. Repeated test inspection is not fresh validation.</p>'
    intro += '<section id="coverage"><h2>Coverage</h2><pre>'+escape(json.dumps(comparison.coverage,indent=2))+'</pre></section>'
    tail = '<section id="dates"><h2>Dates and sensitivity</h2>'+daily+_table(tables['date_sensitivity'])
    tail += '<p><a href="review.json">Configuration and data fingerprint</a> · <a href="summary.csv">Exact summary</a></p></section>'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>'+STYLE+'</style></head><body class="analysis-report"><main>'
    (output/'report.html').write_text(document+intro+''.join(sections)+tail+'</main></body></html>',encoding='utf-8')
    return output
