"""Scoped presentation shared by notebook widgets and portable HTML reviews."""
# SETUP LOGIC: Styling has no data access, model execution, or notebook-global selectors.
from html import escape


# UI LOGIC: Only descendants of this workbench are styled; other notebook outputs retain their theme.
NOTEBOOK_STYLE = """
.analysis-workbench{--ink:#172c45;--muted:#50647a;--line:#dce5ee;--accent:#086d78;
  color:var(--ink);background:#f4f7fb;padding:18px;border:1px solid var(--line);
  border-radius:18px;font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  width:100%;max-width:1440px;box-sizing:border-box;gap:14px;min-width:0}
.analysis-workbench *{box-sizing:border-box}
.analysis-workbench .widget-vbox{gap:12px;min-width:0}
.analysis-workbench .widget-html-content{max-width:100%;line-height:1.55}
.analysis-workbench .analysis-hero{padding:25px 27px;background:linear-gradient(115deg,#132e48,#105765);
  border-radius:13px;color:#fff}
.analysis-workbench .analysis-eyebrow{font-size:11px;font-weight:750;letter-spacing:.16em;text-transform:uppercase;color:#b6e3e7}
.analysis-workbench .analysis-hero h2{color:#fff;font-size:26px;line-height:1.25;margin:7px 0 9px;font-weight:700}
.analysis-workbench .analysis-hero p{margin:0;color:#e1edf5;max-width:900px}
.analysis-workbench .analysis-hero-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}
.analysis-workbench .analysis-hero-tags span{border:1px solid #6e9ca8;border-radius:100px;padding:3px 10px;color:#f4fbff;font-size:12px}
.analysis-workbench .analysis-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;gap:13px;width:100%;min-width:0}
.analysis-workbench .analysis-card-heading h3{font-size:16px;color:var(--ink);margin:0 0 3px;line-height:1.4}
.analysis-workbench .analysis-card-heading p,.analysis-workbench .analysis-help{color:var(--muted);margin:0;font-size:13px;line-height:1.6}
.analysis-workbench .analysis-step{display:inline-flex;align-items:center;justify-content:center;background:#e4f2f2;color:#075965;
  width:27px;height:27px;border-radius:8px;margin-right:8px;font-size:12px;vertical-align:middle}
.analysis-workbench .analysis-row{display:flex;flex-flow:row wrap;align-items:flex-end;gap:12px;width:100%;min-width:0}
.analysis-workbench .analysis-control{min-width:0;max-width:100%;margin:0!important}
.analysis-workbench .analysis-control:not(.widget-checkbox){display:flex;flex-direction:column;align-items:stretch;height:auto}
.analysis-workbench .analysis-control .widget-label{width:100%!important;height:auto;min-height:20px;padding:0;margin:0 0 5px;
  text-align:left;white-space:normal;overflow:visible;text-overflow:clip;font-size:12px;font-weight:650;color:#304960;line-height:1.45}
.analysis-workbench .analysis-control input:not([type=checkbox]),.analysis-workbench .analysis-control select,
.analysis-workbench .analysis-control textarea{width:100%!important;min-width:0;max-width:100%;border:1px solid #c8d5e1;
  border-radius:7px;background:#fff;color:#172c45;padding:7px 9px;min-height:35px;font-size:13px}
.analysis-workbench .analysis-control input:not([type=checkbox]),.analysis-workbench .analysis-control select:not([multiple]){flex:none;height:36px}
.analysis-workbench .analysis-control select:not([multiple]){appearance:none;padding-right:28px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23526a7f' stroke-width='1.7'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 11px center}
.analysis-workbench .analysis-control textarea{flex:1 1 auto;min-height:140px}
.analysis-workbench .analysis-control select[multiple]{flex:none;padding:5px;min-height:125px}
.analysis-workbench .analysis-control select[multiple] option{padding:5px 7px;border-radius:4px}
.analysis-workbench .analysis-control textarea{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.55}
.analysis-workbench .analysis-control input:focus,.analysis-workbench .analysis-control select:focus,
.analysis-workbench .analysis-control textarea:focus{outline:2px solid #0a7c88;outline-offset:2px}
.analysis-workbench .analysis-control.widget-checkbox{min-height:35px;align-self:center}
.analysis-workbench .analysis-control.widget-checkbox label{white-space:normal;overflow:visible;width:100%;line-height:1.5}
.analysis-workbench .analysis-control.widget-checkbox .widget-label{width:auto!important}
.analysis-workbench .widget-button{height:auto;min-height:38px;white-space:normal;border-radius:7px;padding:8px 16px;
  font-size:13px;font-weight:650;box-shadow:none;border:1px solid #c7d5e1;color:#253f57;background:#fff}
.analysis-workbench .widget-button.mod-primary{background:#086d78;color:white;border-color:#086d78}
.analysis-workbench .widget-button.mod-info{background:#eaf4f7;color:#075965;border-color:#b5d4dd}
.analysis-workbench .widget-button:disabled{opacity:.5}
.analysis-workbench .widget-button:focus-visible{outline:3px solid #59aeba;outline-offset:3px}
.analysis-workbench .analysis-actions{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;gap:12px}
.analysis-workbench .analysis-status{padding:12px 14px;border:1px solid #ccdae7;border-left:4px solid #428695;
  border-radius:8px;background:#edf5f9;color:#29485d;margin:0;overflow-wrap:anywhere}
.analysis-workbench .analysis-applied:empty{display:none}
.analysis-workbench .analysis-applied{color:#314e62;font-size:13px;overflow-wrap:anywhere}
.analysis-workbench .analysis-badge{display:inline-flex;padding:5px 10px;border-radius:100px;font-size:12px;font-weight:700;line-height:1.4}
.analysis-workbench .analysis-badge-pending{background:#fff2d7;color:#805100;border:1px solid #ebd296}
.analysis-workbench .analysis-badge-ready{background:#e2f3ec;color:#17604a;border:1px solid #acd7c7}
.analysis-workbench .analysis-badge-neutral{background:#eaf0f6;color:#435d74;border:1px solid #ccd9e5}
.analysis-workbench .analysis-badge-busy{background:#e5f2fa;color:#1b5479;border:1px solid #b0d3e8}
.analysis-workbench .analysis-table-wrap{overflow:auto;max-height:440px;border:1px solid var(--line);border-radius:9px;background:#fff;max-width:100%;margin:10px 0}
.analysis-workbench table{border-collapse:separate;border-spacing:0;font-size:12px;width:100%;font-variant-numeric:tabular-nums;color:#223b51}
.analysis-workbench th,.analysis-workbench td{padding:9px 11px!important;border:0!important;border-bottom:1px solid #e7edf3!important;text-align:right;white-space:nowrap}
.analysis-workbench thead th{position:sticky;top:0;background:#e9f0f6;z-index:1;font-weight:700;color:#29465c}
.analysis-workbench tbody tr:nth-child(even){background:#f5f8fb}
.analysis-workbench tbody tr:hover{background:#eaf4f5}
.analysis-workbench th:first-child,.analysis-workbench td:first-child{text-align:left}
.analysis-workbench .jupyter-widget-Collapse-header .fa{font:inherit;display:inline-block;width:16px}
.analysis-workbench .jupyter-widget-Collapse-header .fa-caret-right::before{content:"›"}
.analysis-workbench .jupyter-widget-Collapse-header .fa-caret-down::before{content:"⌄"}
.analysis-workbench .p-Accordion,.analysis-workbench .lm-Accordion{width:100%;min-width:0}
.analysis-workbench .jupyter-widget-Collapse-header,.analysis-workbench .p-Accordion-header,.analysis-workbench .lm-Accordion-header{font-weight:650;color:#29465c;background:#edf3f8;
  border:1px solid #d5e1eb;border-radius:7px;white-space:normal;height:auto;min-height:36px;padding:9px 12px}
.analysis-workbench .jupyter-widget-Collapse-contents,.analysis-workbench .p-Accordion-child,.analysis-workbench .lm-Accordion-child{padding:12px;border-color:#d5e1eb;background:#fff}
.analysis-workbench .p-TabBar-tab,.analysis-workbench .lm-TabBar-tab{font-size:13px;font-weight:650;padding:8px 14px;
  min-height:38px;height:auto;min-width:85px;color:#3e576d;background:#edf3f8;border-color:#d8e3ec}
.analysis-workbench .p-TabBar-tab.p-mod-current,.analysis-workbench .lm-TabBar-tab.lm-mod-current{color:#075965;background:#fff;border-top:3px solid #086d78}
.analysis-workbench .p-TabBar-content,.analysis-workbench .lm-TabBar-content{overflow-x:auto}
.analysis-workbench .widget-tab-contents{padding:15px;background:#fff;border:1px solid #d8e3ec;min-width:0}
.analysis-workbench .widget-image{max-width:100%;height:auto}
.analysis-workbench .analysis-progress{width:100%;min-width:0}
.analysis-workbench .analysis-progress .widget-label{width:auto!important;min-width:150px;white-space:normal}
@media(max-width:700px){.analysis-workbench{padding:10px;gap:10px}.analysis-workbench .analysis-hero{padding:20px}
 .analysis-workbench .analysis-hero h2{font-size:23px}.analysis-workbench .analysis-card{padding:13px}
 .analysis-workbench .analysis-row>.analysis-control{flex-basis:100%!important}.analysis-workbench .widget-tab-contents{padding:8px}}
"""

# UI LOGIC: The exported report uses the same palette while remaining portable without widget scripts.
REPORT_STYLE = """
.analysis-report{margin:0;background:#f3f6fa;color:#182f46;font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.analysis-report *{box-sizing:border-box}.analysis-report main,.analysis-report .report-shell{max-width:1280px;margin:auto;padding:28px}
.analysis-report h1,.analysis-report h2,.analysis-report h3{line-height:1.25;letter-spacing:-.02em;color:#17334b}
.analysis-report h1{font-size:34px;margin:8px 0 12px}.analysis-report h2{font-size:24px;margin:0 0 14px}.analysis-report h3{font-size:18px}
.analysis-report .hero,.analysis-report header{background:linear-gradient(115deg,#132e48,#105765);color:#e7f3f8;border-radius:16px;padding:32px;margin:0 0 22px}
.analysis-report .hero h1,.analysis-report header h1{color:#fff}.analysis-report .eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#b6e3e7;font-weight:700}
.analysis-report section,.analysis-report .card{background:white;border:1px solid #dbe5ee;border-radius:13px;padding:24px;margin:20px 0;min-width:0}
.analysis-report nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.analysis-report nav a{display:inline-block;background:#fff;border:1px solid #cddde7;border-radius:100px;padding:7px 14px;text-decoration:none;font-size:13px;font-weight:650;color:#075965}
.analysis-report nav a:hover{background:#e6f1f3}.analysis-report a{color:#006a78}.analysis-report a:focus-visible{outline:3px solid #147e8b;outline-offset:3px}
.analysis-report .muted,.analysis-report .note{color:#52697f}.analysis-report .callout{background:#edf6f8;border-left:4px solid #217d88;padding:14px 18px;border-radius:7px}
.analysis-report .metrics,.analysis-report .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.analysis-report .metric{border:1px solid #d9e6ee;border-radius:10px;padding:17px;background:#f7fafc}.analysis-report .metric strong{display:block;font-size:25px;color:#0c606d}
.analysis-report .table-wrap,.analysis-report .table-scroll{max-width:100%;overflow:auto;max-height:600px;border:1px solid #dce5ee;border-radius:9px;margin:16px 0}
.analysis-report table{width:100%;border-collapse:separate;border-spacing:0;font-size:12px;font-variant-numeric:tabular-nums}
.analysis-report th,.analysis-report td{padding:10px 12px;text-align:right;white-space:nowrap;border:0;border-bottom:1px solid #e3ebf2}
.analysis-report thead th{position:sticky;top:0;background:#eaf1f6;color:#29465c;z-index:1}.analysis-report tbody tr:nth-child(even){background:#f5f8fb}
.analysis-report th:first-child,.analysis-report td:first-child{text-align:left}.analysis-report tbody tr:hover{background:#eaf4f5}
.analysis-report img{max-width:100%;height:auto;display:block;margin:14px auto}.analysis-report details{border:1px solid #dce5ee;border-radius:9px;padding:12px 16px;margin:12px 0}
.analysis-report summary{cursor:pointer;font-weight:650;color:#255267}.analysis-report code{font-size:.88em;background:#eaf0f5;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}
.analysis-report .badge{display:inline-block;padding:4px 10px;border-radius:100px;background:#e2f3ec;color:#17604a;font-size:12px;font-weight:700}
@media(max-width:700px){.analysis-report main,.analysis-report .report-shell{padding:12px}.analysis-report .hero,.analysis-report header{padding:23px}.analysis-report h1{font-size:27px}.analysis-report section,.analysis-report .card{padding:16px}}
@media print{.analysis-report{background:#fff}.analysis-report nav{display:none}.analysis-report section{break-inside:avoid}.analysis-report .table-wrap,.analysis-report .table-scroll{max-height:none;overflow:visible}}
"""


def control(widget, *, wide=False):
    # UI LOGIC: Native labels stay attached to inputs; stacked labels retain their complete text.
    widget.add_class('analysis-control')
    widget.style.description_width = 'initial'
    widget.layout.width = 'auto'
    widget.layout.min_width = '0'
    widget.layout.flex = '1 1 420px' if wide else '1 1 230px'
    if getattr(widget, 'description', ''):
        widget.tooltip = widget.description.rstrip(':')
    return widget


def row(*children):
    # UI LOGIC: Wrapping works at the notebook pane width, independently of the browser viewport.
    import ipywidgets as w
    return w.Box(list(children), layout=w.Layout(display='flex', flex_flow='row wrap',
                 align_items='flex-end', width='100%')).add_class('analysis-row')


def section(title, *children, note='', step=None):
    # UI LOGIC: Escape all dynamic copy before inserting the card's descriptive heading.
    import ipywidgets as w
    number = f'<span class="analysis-step">{escape(str(step))}</span>' if step is not None else ''
    heading = w.HTML(f'<div class="analysis-card-heading"><h3>{number}{escape(title)}</h3><p>{escape(note)}</p></div>')
    return w.VBox([heading, *children], layout=w.Layout(width='100%')).add_class('analysis-card')


def disclosure(title, *children, opened=False):
    # UI LOGIC: Advanced controls retain their values when the containing disclosure is collapsed.
    import ipywidgets as w
    widget = w.Accordion(children=[w.VBox(list(children), layout=w.Layout())],
                         selected_index=0 if opened else None)
    widget.set_title(0, title)
    return widget


def hero(title, description, *, eyebrow='Prediction research', tags=()):
    # UI LOGIC: Compact context precedes the controls; tags describe the current workflow, never computed claims.
    import ipywidgets as w
    labels = ''.join(f'<span>{escape(str(tag))}</span>' for tag in tags)
    return w.HTML(f'<div class="analysis-hero"><div class="analysis-eyebrow">{escape(eyebrow)}</div>'
                  f'<h2>{escape(title)}</h2><p>{escape(description)}</p><div class="analysis-hero-tags">{labels}</div></div>')


def badge(label, state='neutral'):
    # UI LOGIC: Both words and color identify state so the distinction does not depend on color perception.
    return f'<span class="analysis-badge analysis-badge-{escape(state)}" role="status">{escape(label)}</span>'


def table_html(table, *, title='', limit=100):
    # UI LOGIC: Bound notebook rows while preserving the full table in the caller's public result.
    shown = table.head(limit)
    heading = f'<h4>{escape(title)}</h4>' if title else ''
    note = f'<p class="analysis-help">Showing {len(shown):,} of {len(table):,} rows. Complete values are available in the export.</p>'
    return heading+note+'<div class="analysis-table-wrap" tabindex="0" role="region" aria-label="Scrollable results table">'+shown.to_html(
        index=False, escape=True, border=0, float_format=lambda value:f'{value:.5g}')+'</div>'


def style_root(widget):
    # UI LOGIC: Each independently displayed form carries its own scoped style, including nested reviews.
    import ipywidgets as w
    widget.add_class('analysis-workbench')
    widget.layout.width = '100%'
    widget.layout.min_width = '0'
    stylesheet = w.HTML('<style>'+NOTEBOOK_STYLE+'</style>',layout=w.Layout(display='none'))
    widget.children = (stylesheet, *widget.children)
    return widget
