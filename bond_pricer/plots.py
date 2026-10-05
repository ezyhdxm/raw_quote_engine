"""Complete notebook figures for pairwise model review, including support-aware heatmaps."""
# SETUP LOGIC: Unmanaged figures avoid duplicate notebook displays and global plot state.
import numpy as np
from matplotlib.figure import Figure

# CONFIGURATION LOGIC: Use reader-facing metric names while keeping stable machine-readable table columns.
METRICS = {'mae_delta':'MAE change (candidate minus reference)', 'mae_improvement_pct':'Relative MAE improvement (%)',
           'p95_delta':'P95 error change (candidate minus reference)', 'candidate_mae':'Candidate MAE',
           'candidate_rmse':'Candidate RMSE', 'candidate_p95':'Candidate P95 absolute error', 'win_rate_pct':'Paired win rate (%)'}


def overview(comparison):
    # PLOTTING LOGIC: Display common-sample losses and coverage with explicit model names and units.
    table = comparison.summary().iloc[0]
    fig = Figure(figsize=(12,4.5), constrained_layout=True)
    axes = fig.subplots(1,3)
    names = [comparison.reference_name, comparison.candidate_name]
    colors = ['#657987','#187f83']
    for ax, metric, title in zip(axes[:2], ['mae','p95'], ['Mean absolute error','95th percentile absolute error']):
        values = [table[f'reference_{metric}'], table[f'candidate_{metric}']]
        ax.bar(names, values, color=colors)
        ax.set(title=title, ylabel=comparison.unit)
        ax.tick_params(axis='x', rotation=15)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f'{value:.4g}', (i,value), xytext=(0,4), textcoords='offset points', ha='center')
    counts = comparison.coverage
    axes[2].bar(['Supplied','Valid target','Paired'], [counts[k] for k in ['total','valid_target','paired']], color='#426b81')
    axes[2].set(title='Evaluation coverage', ylabel='Trades')
    fig.suptitle(f'{comparison.candidate_name} vs {comparison.reference_name} | {counts["paired"]:,} paired trades')
    return fig


def slice_figure(table, metric='mae_delta', *, unit='bps', title='Slice comparison'):
    # PLOTTING LOGIC: Retain every observed group, flag small samples and pair losses with support.
    height = min(24, max(4, len(table)*.3+1.8))
    fig = Figure(figsize=(12,height), constrained_layout=True)
    ax, support = fig.subplots(1,2, gridspec_kw={'width_ratios':[3,1]})
    labels = [str(v)+(' *' if small else '') for v,small in zip(table['group'],table['low_support'])]
    y = np.arange(len(table))
    values = table[metric].to_numpy(dtype=float)
    beneficial = values >= 0 if metric == 'mae_improvement_pct' else values <= 0
    colors = np.where(beneficial, '#187f83','#c67446') if metric in {'mae_delta','p95_delta','mae_improvement_pct'} else '#187f83'
    ax.barh(y, values, color=colors)
    ax.axvline(0,color='#293d4a',linewidth=.8)
    ax.set(yticks=y, yticklabels=labels, title=METRICS.get(metric,metric), xlabel='%' if metric.endswith('pct') else unit)
    support.barh(y,table['n'],color='#657987')
    support.set(yticks=y,yticklabels=[],title='Paired trades',xlabel='N')
    ax.invert_yaxis()
    support.invert_yaxis()
    fig.suptitle(title+'\n* below minimum support; negative error delta = improvement')
    return fig


def heatmap(table, metric='mae_delta', *, unit='bps', title='Two-column comparison'):
    # CORE LOGIC: STEP 1 — Align metric and sample-count grids to the same observed category order.
    # Input: (x,y,delta,n)=[('A','big',-1,5),('A','small',2,3),('B','big',-.5,2)].
    # Output: metric rows A/B, columns big/small = [[-1,2],[-.5,NaN]]; counts=[[5,3],[2,NaN]].
    # Explanation: Missing B/small remains empty in both panels instead of being filled with zero.
    # Trick: Both pivots are explicitly reindexed; categorical order cannot swap cells between panels.
    xs, ys = table['x'].drop_duplicates().tolist(), table['y'].drop_duplicates().tolist()
    grid = table.pivot(index='x',columns='y',values=metric).reindex(index=xs,columns=ys)
    counts = table.pivot(index='x',columns='y',values='n').reindex(index=xs,columns=ys)
    low = table.pivot(index='x',columns='y',values='low_support').reindex(index=xs,columns=ys)
    # PLOTTING LOGIC: Use a symmetric difference scale and a separate support panel, with no 3D occlusion.
    fig = Figure(figsize=(max(12,len(ys)*.9),max(5,len(xs)*.42)), constrained_layout=True)
    axes = fig.subplots(1,2)
    values = grid.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0, 1e-9)
    signed = metric in {'mae_delta','p95_delta','mae_improvement_pct','reference_bias','candidate_bias'}
    cmap = ('RdBu' if metric == 'mae_improvement_pct' else 'RdBu_r') if signed else 'Blues'
    images = [axes[0].imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=-bound if signed else 0,vmax=bound),
              axes[1].imshow(np.ma.masked_invalid(counts.to_numpy(dtype=float)),aspect='auto',cmap='Blues')]
    for ax, image, label in zip(axes,images,[METRICS.get(metric,metric),'Paired trades']):
        ax.set(xticks=range(len(ys)),xticklabels=[str(v) for v in ys],yticks=range(len(xs)),yticklabels=[str(v) for v in xs],title=label)
        ax.tick_params(axis='x',rotation=45)
        fig.colorbar(image,ax=ax,shrink=.7)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if np.isfinite(values[i,j]):
                star = '*' if low.iloc[i,j] else ''
                axes[0].text(j,i,f'{values[i,j]:.3g}{star}',ha='center',va='center',fontsize=8,
                             color='white' if abs(values[i,j]) > .55*bound else 'black')
            if np.isfinite(counts.iloc[i,j]):
                axes[1].text(j,i,f'{counts.iloc[i,j]:,.0f}',ha='center',va='center',fontsize=8,
                             color='white' if images[1].norm(counts.iloc[i,j]) > .55 else 'black')
    fig.suptitle(title+f' | {"%" if metric.endswith("pct") else unit}\n* below minimum support; blank = undefined / unavailable metric; see N')
    return fig


def daily_figure(table, *, unit='bps', title='Daily mean absolute errors'):
    # PLOTTING LOGIC: Date labels come from the comparison timezone; no fixed time window is assumed.
    fig = Figure(figsize=(12,5), constrained_layout=True)
    ax = fig.subplots()
    x = np.arange(len(table))
    ax.plot(x,table['reference_mae'],label='Reference',color='#657987',marker='.')
    ax.plot(x,table['candidate_mae'],label='Candidate',color='#187f83',marker='.')
    step = max(1,len(table)//15)
    ax.set(xticks=x[::step],xticklabels=table['group'].astype(str).iloc[::step],ylabel=unit,title=title)
    ax.tick_params(axis='x',rotation=35)
    ax.legend()
    return fig
