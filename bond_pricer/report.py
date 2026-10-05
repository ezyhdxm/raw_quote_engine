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

STYLE = '''body{max-width:1120px;margin:35px auto;padding:0 22px;font:16px/1.65 system-ui;color:#183541}
h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #d5e0e4;text-align:left}
th{background:#edf4f5}.table{overflow:auto}img{max-width:100%}.note{background:#edf7f5;padding:16px;border-left:4px solid #18807d}
code{overflow-wrap:anywhere}a{color:#097986}section{margin:32px 0}details{margin:15px 0}'''


def _table(frame):
    # FORMATTING LOGIC: Exact machine-readable values are exported separately; HTML is rounded for reading.
    return '<div class="table">'+frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.5g}')+'</div>'


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
    plots.overview(comparison).savefig(output/'overview.png',dpi=160)
    plots.daily_figure(tables['daily'],unit=comparison.unit).savefig(output/'daily.png',dpi=160)
    sections = ['<h2>Common-sample overview</h2>'+_table(tables['summary'])+'<img src="overview.png" alt="Common-sample metrics and coverage">']
    # REPORTING LOGIC: Build every requested slice from the same result object used in the notebook.
    for i,spec in enumerate(specs):
        key = f'slice_{i+1:02d}'
        table = comparison.slice(spec,min_count=min_count)
        tables[key] = table
        title = spec.name or spec.column
        plots.slice_figure(table,metric,unit=comparison.unit,title=title).savefig(output/f'{key}.png',dpi=160)
        sections.append(f'<h2>{escape(title)}</h2><img src="{key}.png" alt="Slice loss and support"><details><summary>Complete table</summary>'+_table(table)+'</details>')
    # REPORTING LOGIC: Cross tables preserve every observed cell and its support.
    for i,pair in enumerate(interactions):
        sx,sy = [s if isinstance(s,Slice) else Slice(s) for s in pair]
        key = f'interaction_{i+1:02d}'
        table = comparison.cross_slice(sx,sy,min_count=min_count)
        tables[key] = table
        if not table.empty:
            plots.heatmap(table,metric,unit=comparison.unit,title=f'{sx.column} × {sy.column}').savefig(output/f'{key}.png',dpi=160)
            sections.append(f'<h2>{escape(sx.column)} × {escape(sy.column)}</h2><img src="{key}.png" alt="Loss and count heatmaps">'+_table(table))
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
    intro = f'<h1>{escape(title)}</h1><p>Errors in {escape(comparison.unit)}. Record-weighted metrics on common finite targets and predictions.</p>'
    intro += '<p class="note">Negative MAE/P95 delta means improvement; positive MAE improvement % means improvement. '
    intro += 'Sparse cells are flagged, never removed. Slices are descriptive, overlap, and must not be added together. '
    intro += 'Date sensitivity is not a confidence interval. Repeated test inspection is not fresh validation.</p>'
    intro += '<h2>Coverage</h2><pre>'+escape(json.dumps(comparison.coverage,indent=2))+'</pre>'
    tail = '<h2>Dates and sensitivity</h2><img src="daily.png" alt="Daily losses">'+_table(tables['date_sensitivity'])
    tail += '<p><a href="review.json">Configuration and data fingerprint</a> · <a href="summary.csv">Exact summary</a></p>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+escape(title)+'</title><style>'+STYLE+'</style><body>'
    (output/'report.html').write_text(document+intro+''.join('<section>'+s+'</section>' for s in sections)+tail+'</body></html>',encoding='utf-8')
    return output
