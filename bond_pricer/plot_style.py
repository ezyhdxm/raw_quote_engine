"""Local figure styling and measured label layout for notebook and report figures."""
# SETUP LOGIC: Agg measures text without registering pyplot figures or changing rcParams.
import textwrap
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter, ConciseDateFormatter, AutoDateFormatter, DateLocator
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# CONFIGURATION LOGIC: Shared semantic colors keep favorable changes and support consistent.
TEAL = '#187f83'
SLATE = '#657987'
ORANGE = '#c67446'
INK = '#263747'
GRID = '#e8edf2'


def wrapped_label(value, width=24):
    # PLOTTING LOGIC: Preserve full names, including distinguishing suffixes in long identifiers.
    return '\n'.join(textwrap.fill(line, width=width, break_long_words=True, break_on_hyphens=False)
                     for line in str(value).split('\n'))


def make_figure(figsize=(12, 5)):
    # PLOTTING LOGIC: An unmanaged white figure is independent of the notebook's active plotting theme.
    figure = Figure(figsize=figsize, facecolor='white', constrained_layout=True)
    figure.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.07, hspace=.10)
    return figure


def figure_title(figure, title, subtitle=''):
    # PLOTTING LOGIC: Titles use a consistent reading hierarchy; long model names remain complete.
    text = wrapped_label(title, max(65, int(figure.get_figwidth() * 8)))
    figure.suptitle(text, x=.02, ha='left', fontsize=14, fontweight='semibold', color=INK,
                   fontfamily='DejaVu Sans', linespacing=1.5)
    if subtitle:
        figure.supxlabel(wrapped_label(subtitle, max(75, int(figure.get_figwidth() * 12))),
                         fontsize=9, color=SLATE, fontfamily='DejaVu Sans', linespacing=1.5)


def _plain_numeric_axis(axis, name):
    # PLOTTING LOGIC: Date and nonlinear coordinates keep their specialized tick semantics.
    coordinate = getattr(axis, name + 'axis')
    date_formatter = isinstance(coordinate.get_major_formatter(), (DateFormatter, ConciseDateFormatter, AutoDateFormatter))
    return (getattr(axis, f'get_{name}scale')() == 'linear' and not date_formatter
            and not isinstance(coordinate.get_major_locator(), DateLocator))


def style_axis(axis, grid='y'):
    # PLOTTING LOGIC: Mutate only the supplied axes, using subtle grids and readable numeric notation.
    axis.set_facecolor('white')
    axis.set_axisbelow(True)
    for side, spine in axis.spines.items():
        spine.set_visible(side in {'bottom', 'left'} and not axis.images)
        spine.set_color('#d4dee7')
        spine.set_linewidth(.7)
    axis.grid(False)
    if grid and not axis.images:
        axis.grid(True, axis=grid, color=GRID, linewidth=.7)
    axis.tick_params(axis='both', colors=SLATE, labelsize=9, length=0, pad=7)
    axis.title.set(fontsize=11, fontweight='semibold', color=INK, fontfamily='DejaVu Sans')
    axis.title.set_text(wrapped_label(axis.get_title(), 48))
    axis.xaxis.label.set(fontsize=10, color=SLATE, fontfamily='DejaVu Sans')
    axis.yaxis.label.set(fontsize=10, color=SLATE, fontfamily='DejaVu Sans')
    for coordinate in (axis.xaxis, axis.yaxis):
        if (not getattr(axis, '_comparison_category_' + coordinate.axis_name, False)
                and _plain_numeric_axis(axis, coordinate.axis_name)):
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_powerlimits((-3, 5))
            coordinate.set_major_formatter(formatter)
            coordinate.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=2))
        coordinate.get_offset_text().set(fontsize=8, color=SLATE)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontfamily('DejaVu Sans')
    legend = axis.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for label in legend.get_texts():
            label.set(fontsize=9, color=INK, fontfamily='DejaVu Sans')


def categorical_ticks(ax, labels, axis_name='x', width=18, **kwargs):
    # PLOTTING LOGIC: Keep all categories visible and wrap names instead of truncating their identity.
    axis_name = kwargs.pop('axis', axis_name)
    labels = [wrapped_label(value, width) for value in labels]
    getattr(ax, f'set_{axis_name}ticks')(np.arange(len(labels)), labels)
    setattr(ax, '_comparison_category_' + axis_name, True)
    ax.tick_params(axis=axis_name, labelrotation=0)
    return labels


def sequence_ticks(ax, labels, axis_name='x', max_ticks=9, **kwargs):
    # PLOTTING LOGIC: An ordered sequence keeps first/last dates and a measured subset of interior ticks.
    # Trick: The positions are observation indices, so missing calendar dates do not manufacture data points.
    axis_name = kwargs.pop('axis', axis_name)
    labels = [str(value) for value in labels]
    positions = np.unique(np.linspace(0, len(labels)-1, min(len(labels), max_ticks), dtype=int))
    getattr(ax, f'set_{axis_name}ticks')(positions, [labels[index] for index in positions])
    setattr(ax, '_comparison_category_' + axis_name, True)
    setattr(ax, '_comparison_sequence_' + axis_name, labels)
    ax.tick_params(axis=axis_name, labelrotation=0)


def row_figure_height(labels, width=32, minimum=4.7):
    # PLOTTING LOGIC: Allocate physical space for every wrapped row instead of shrinking a large category set.
    lines = sum(max(1, wrapped_label(label, width).count('\n') + 1) for label in labels)
    return max(minimum, 2.0 + .16 * lines + .10 * len(labels))


def heatmap_size(rows, columns, *, cell_width=.66, cell_height=.48, row_width=28, column_width=10, stacked=False):
    # PLOTTING LOGIC: Cell annotations and multi-line category labels both receive real drawing space.
    row_lines = max((wrapped_label(value, row_width).count('\n') + 1 for value in rows), default=1)
    column_lines = max((wrapped_label(value, column_width).count('\n') + 1 for value in columns), default=1)
    width = max(13, 2 * (len(columns) * cell_width + 3.5))
    height = max(5.5, len(rows) * max(cell_height, row_lines * .15) + 2.7 + column_lines * .15)
    return (max(12, width / 2), height * 2 - 1.5) if stacked else (width, height)


def _overlap_ratio(labels, renderer, axis_name):
    # PLOTTING LOGIC: Compare actual rendered tick boxes, including multiline labels and scientific notation.
    boxes = [label.get_window_extent(renderer) for label in labels if label.get_visible() and label.get_text()]
    boxes.sort(key=lambda box: box.x0 if axis_name == 'x' else box.y0)
    ratios = []
    for first, second in zip(boxes, boxes[1:]):
        first_center = (first.x0 + first.x1) / 2 if axis_name == 'x' else (first.y0 + first.y1) / 2
        second_center = (second.x0 + second.x1) / 2 if axis_name == 'x' else (second.y0 + second.y1) / 2
        required = (first.width + second.width) / 2 + 9 if axis_name == 'x' else (first.height + second.height) / 2 + 5
        ratios.append(required / max(second_center - first_center, 1))
    return max(ratios, default=0)


def _fit_axis_ticks(axis, coordinate, renderer):
    # PLOTTING LOGIC: Thin date/numeric ticks before enlarging categorical axes; category values stay complete.
    labels = getattr(axis, f'get_{coordinate}ticklabels')()
    ratio = _overlap_ratio(labels, renderer, coordinate)
    if ratio <= 1:
        return 1.0
    sequence = getattr(axis, '_comparison_sequence_' + coordinate, None)
    if sequence is not None and len(labels) > 2:
        count = max(2, min(len(labels)-1, int(len(labels) / ratio)))
        sequence_ticks(axis, sequence, axis_name=coordinate, max_ticks=count)
        return 1.0
    if not getattr(axis, '_comparison_category_' + coordinate, False):
        if not _plain_numeric_axis(axis, coordinate):
            return 1.0
        count = max(2, int(len(labels) / ratio) - 1)
        getattr(axis, coordinate + 'axis').set_major_locator(MaxNLocator(nbins=count, min_n_ticks=2))
        return 1.0
    return min(ratio * 1.08, 2.0)


def _cell_text_scale(axis, renderer):
    # PLOTTING LOGIC: Cell counts, t/q values, and multiline support notes must fit their own heatmap cells.
    if not axis.images or not axis.texts:
        return 1.0, 1.0
    corners = axis.transData.transform([[-.5, -.5], [.5, .5]])
    width, height = np.abs(corners[1] - corners[0])
    boxes = [text.get_window_extent(renderer) for text in axis.texts if text.get_visible()]
    return (max([1.0] + [box.width * 1.10 / max(width, 1) for box in boxes]),
            max([1.0] + [box.height * 1.15 / max(height, 1) for box in boxes]))


def finish_figure(figure):
    # PLOTTING LOGIC: Finalize local styling and verify tick geometry with the same renderer used for PNG export.
    for axis in figure.axes:
        if getattr(axis, '_colorbar', None) is None:
            style_axis(axis, grid='x' if getattr(axis, '_comparison_category_y', False) else 'y')
        else:
            axis.tick_params(labelsize=8, colors=SLATE, length=0)
            axis.spines['outline'].set_visible(False)
    canvas = FigureCanvasAgg(figure)
    for _ in range(5):
        canvas.draw()
        renderer = canvas.get_renderer()
        width_scale, height_scale = 1.0, 1.0
        before = [(len(axis.get_xticks()), len(axis.get_yticks())) for axis in figure.axes]
        for axis in figure.axes:
            if getattr(axis, '_colorbar', None) is None:
                width_scale = max(width_scale, _fit_axis_ticks(axis, 'x', renderer))
                height_scale = max(height_scale, _fit_axis_ticks(axis, 'y', renderer))
                cell_width, cell_height = _cell_text_scale(axis, renderer)
                width_scale, height_scale = max(width_scale, cell_width), max(height_scale, cell_height)
        if width_scale > 1 or height_scale > 1:
            figure.set_size_inches(figure.get_figwidth()*width_scale, figure.get_figheight()*height_scale)
        elif before == [(len(axis.get_xticks()), len(axis.get_yticks())) for axis in figure.axes]:
            break
    canvas.draw()
    return figure
