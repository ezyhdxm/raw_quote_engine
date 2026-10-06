"""Readable comparison figures with complete labels and support-aware heatmaps."""
# SETUP LOGIC: All styling is local to the returned unmanaged Figure.
import numpy as np
from .plot_style import (TEAL, SLATE, ORANGE, INK, make_figure, finish_figure, figure_title,
                         categorical_ticks, sequence_ticks, row_figure_height, heatmap_size)

# CONFIGURATION LOGIC: Reader-facing metric names retain stable machine-readable table columns.
METRICS = {'mae_delta': 'MAE change (candidate minus reference)', 'mae_improvement_pct': 'Relative MAE improvement (%)',
           'p95_delta': 'P95 error change (candidate minus reference)', 'candidate_mae': 'Candidate MAE',
           'candidate_rmse': 'Candidate RMSE', 'candidate_p95': 'Candidate P95 absolute error', 'win_rate_pct': 'Paired win rate (%)'}


def overview(comparison):
    # PLOTTING LOGIC: Display common-sample losses, explicit model identities, and the evaluation coverage funnel.
    table = comparison.summary().iloc[0]
    fig = make_figure((14, 6.2))
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.35]})
    names = [comparison.reference_name, comparison.candidate_name]
    for ax, metric, title in zip(axes[:2], ['mae', 'p95'], ['Mean absolute error', '95th percentile absolute error']):
        values = [table[f'reference_{metric}'], table[f'candidate_{metric}']]
        ax.bar(np.arange(2), values, color=[SLATE, TEAL], width=.58, zorder=3)
        categorical_ticks(ax, names, width=22)
        ax.set(title=title, ylabel=comparison.unit)
        ax.margins(y=.20)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f'{value:.4g}', (i, value), xytext=(0, 7), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='semibold', color=INK)
    counts = comparison.coverage
    labels = ['Supplied', 'Valid target', 'Paired']
    keys = ['total', 'valid_target', 'paired']
    axes[2].bar(np.arange(len(keys)), [counts[k] for k in keys], color=[SLATE]*2 + [TEAL], width=.65, zorder=3)
    categorical_ticks(axes[2], labels, width=10)
    axes[2].margins(y=.20)
    for i, key in enumerate(keys):
        axes[2].annotate(f'{counts[key]:,}', (i, counts[key]), xytext=(0, 7), textcoords='offset points',
                         ha='center', fontsize=9, color=INK)
    axes[2].set(title='Evaluation coverage', ylabel='Trades')
    figure_title(fig, f'{comparison.candidate_name} vs {comparison.reference_name}',
                 f'{counts["paired"]:,} paired trades · Both error panels use the same paired sample')
    return finish_figure(fig)


def slice_figure(table, metric='mae_delta', *, unit='bps', title='Slice comparison'):
    # PLOTTING LOGIC: Retain every observed group and give wrapped labels physical space at a readable font size.
    labels = [str(value) + (' *' if small else '') for value, small in zip(table['group'], table['low_support'])]
    fig = make_figure((13, row_figure_height(labels)))
    ax, support = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    y = np.arange(len(table))
    values = table[metric].to_numpy(dtype=float)
    beneficial = values >= 0 if metric == 'mae_improvement_pct' else values <= 0
    colors = np.where(beneficial, TEAL, ORANGE) if metric in {'mae_delta', 'p95_delta', 'mae_improvement_pct'} else TEAL
    ax.barh(y, values, color=colors, height=.64, zorder=3)
    ax.axvline(0, color=SLATE, linewidth=.8, zorder=2)
    categorical_ticks(ax, labels, axis='y', width=32)
    ax.set(title=METRICS.get(metric, metric), xlabel='%' if metric.endswith('pct') else unit)
    support.barh(y, table['n'], color=SLATE, height=.64, zorder=3)
    categorical_ticks(support, ['']*len(labels), axis='y')
    support.set(title='Paired trades', xlabel='N')
    for axis in (ax, support):
        axis.invert_yaxis()
        axis.margins(y=.02, x=.08)
    figure_title(fig, title, '* below minimum support · Negative error delta = improvement')
    return finish_figure(fig)


def heatmap(table, metric='mae_delta', *, unit='bps', title='Two-column comparison'):
    # CORE LOGIC: STEP 1 — Align metric and sample-count grids to the same observed category order.
    # Input: (x,y,delta,n)=[('A','big',-1,5),('A','small',2,3),('B','big',-.5,2)].
    # Output: metric rows A/B, columns big/small = [[-1,2],[-.5,NaN]]; counts=[[5,3],[2,NaN]].
    # Explanation: Missing B/small remains empty in both panels instead of being filled with zero.
    # Trick: Both pivots are explicitly reindexed; categorical order cannot swap cells between panels.
    xs, ys = table['x'].drop_duplicates().tolist(), table['y'].drop_duplicates().tolist()
    grid = table.pivot(index='x', columns='y', values=metric).reindex(index=xs, columns=ys)
    counts = table.pivot(index='x', columns='y', values='n').reindex(index=xs, columns=ys)
    low = table.pivot(index='x', columns='y', values='low_support').reindex(index=xs, columns=ys)
    # PLOTTING LOGIC: Symmetric difference scales and a separate support panel keep effect and evidence distinct.
    stacked = len(ys) > 8
    fig = make_figure(heatmap_size(xs, ys, stacked=stacked))
    axes = fig.subplots(2, 1) if stacked else fig.subplots(1, 2)
    figure_title(fig, title + ' | blank = undefined', f'{"%" if metric.endswith("pct") else unit} · * below minimum support · '
                 'Blank = undefined / unavailable metric; see paired N')
    if table.empty:
        for ax in axes:
            ax.text(.5, .5, 'No observed groups', transform=ax.transAxes, ha='center', color=SLATE)
            ax.set_axis_off()
        return finish_figure(fig)
    values = grid.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0, 1e-9)
    signed = metric in {'mae_delta', 'p95_delta', 'mae_improvement_pct', 'reference_bias', 'candidate_bias'}
    cmap = ('RdBu' if metric == 'mae_improvement_pct' else 'RdBu_r') if signed else 'Blues'
    images = [axes[0].imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, vmin=-bound if signed else 0, vmax=bound),
              axes[1].imshow(np.ma.masked_invalid(counts.to_numpy(dtype=float)), aspect='auto', cmap='Blues', vmin=0)]
    for ax, image, label in zip(axes, images, [METRICS.get(metric, metric), 'Paired trades']):
        categorical_ticks(ax, ys, width=10)
        categorical_ticks(ax, xs, axis='y', width=28)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(ys)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(xs)+1)-.5, minor=True)
        ax.tick_params(which='minor', length=0)
        fig.colorbar(image, ax=ax, shrink=.70, fraction=.035, pad=.03)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if np.isfinite(values[i, j]):
                marker = '*' if low.iloc[i, j] else ''
                axes[0].text(j, i, f'{values[i,j]:.3g}{marker}', ha='center', va='center', fontsize=8,
                             color='white' if abs(values[i, j]) > .55*bound else INK)
            if np.isfinite(counts.iloc[i, j]):
                axes[1].text(j, i, f'{counts.iloc[i,j]:,.0f}', ha='center', va='center', fontsize=8,
                             color='white' if images[1].norm(counts.iloc[i, j]) > .55 else INK)
    return finish_figure(fig)


def daily_figure(table, *, unit='bps', title='Daily mean absolute errors'):
    # PLOTTING LOGIC: Tick positions preserve observed-day order and always retain the first and last date.
    fig = make_figure((12, 5))
    ax = fig.subplots()
    x = np.arange(len(table))
    marker = 'o' if len(table) <= 60 else None
    ax.plot(x, table['reference_mae'], label='Reference', color=SLATE, marker=marker, markersize=3, linewidth=1.6)
    ax.plot(x, table['candidate_mae'], label='Candidate', color=TEAL, marker=marker, markersize=3, linewidth=1.9)
    sequence_ticks(ax, table['group'], max_ticks=9)
    ax.set(ylabel=unit, xlabel='Observed date')
    ax.margins(x=.025, y=.12)
    ax.legend(loc='upper left', ncols=2)
    figure_title(fig, title, 'Daily means on the paired sample · Dates follow the comparison timezone')
    return finish_figure(fig)
