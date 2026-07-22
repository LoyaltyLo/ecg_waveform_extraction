"""Generate an interactive HTML dashboard for browsing P-QRS-T waveform results.

Reads all summary.json files from the output directory and produces a
single self-contained index.html with sortable tables and embedded images.
"""

import json
import os
from pathlib import Path


def _read_summary(rec_dir: str) -> dict | None:
    """Read summary.json from a record directory, return None if missing."""
    sp = os.path.join(rec_dir, 'summary.json')
    if not os.path.exists(sp):
        return None
    with open(sp, 'r', encoding='utf-8') as f:
        return json.load(f)


def _count_images(beats_dir: str) -> int:
    """Count waveform PNGs in a beats directory."""
    if not os.path.isdir(beats_dir):
        return 0
    return sum(1 for f in os.listdir(beats_dir) if f.endswith('_waveform.png'))


def build_dashboard(output_dir: str, title: str = "ECG Waveform Dashboard") -> str:
    """Build an HTML dashboard for all records in the output directory.

    Parameters
    ----------
    output_dir : str
        Path to the output directory containing per-record subdirectories.
    title : str
        Page title.

    Returns
    -------
    str : Path to the generated index.html file.
    """
    records = []
    for entry in sorted(os.listdir(output_dir)):
        rec_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(rec_dir):
            continue
        summary = _read_summary(rec_dir)
        if summary is None:
            continue

        lead_i = summary.get('lead_I', {}) or {}
        lead_ii = summary.get('lead_II', {}) or {}
        meas = summary.get('measurements', {}) or {}

        # Count images per lead
        i_plots = _count_images(os.path.join(rec_dir, 'lead_I', 'beats'))
        ii_plots = _count_images(os.path.join(rec_dir, 'lead_II', 'beats'))

        # Check which images exist
        has_seg_i = os.path.exists(os.path.join(rec_dir, 'lead_I', 'segmentation.png'))
        has_seg_ii = os.path.exists(os.path.join(rec_dir, 'lead_II', 'segmentation.png'))

        record = {
            'name': summary.get('record', entry),
            'i_beats': lead_i.get('n_beats', 0),
            'ii_beats': lead_ii.get('n_beats', 0),
            'i_plots': i_plots,
            'ii_plots': ii_plots,
            'i_pwaves': lead_i.get('n_p_waves', 0),
            'ii_pwaves': lead_ii.get('n_p_waves', 0),
            'hr': meas.get('HR') or meas.get('HR_bpm'),
            'p_axis': meas.get('P_axis'),
            'qrs_axis': meas.get('QRS_axis'),
            'qrs_dur': meas.get('QRS_dur'),
            'p_dur': meas.get('P_dur'),
            'interpretation': summary.get('measurements', {}).get('interpretation', ''),
            'has_seg_i': has_seg_i,
            'has_seg_ii': has_seg_ii,
        }
        records.append(record)

    if not records:
        return _write_html(output_dir, [], title)

    # Compute aggregates
    total_i_beats = sum(r['i_beats'] for r in records)
    total_ii_beats = sum(r['ii_beats'] for r in records)
    total_i_plots = sum(r['i_plots'] for r in records)
    total_ii_plots = sum(r['ii_plots'] for r in records)
    hrs = [r['hr'] for r in records if r['hr'] is not None]
    p_axes = [r['p_axis'] for r in records if r['p_axis'] is not None]
    qrs_axes = [r['qrs_axis'] for r in records if r['qrs_axis'] is not None]

    aggregates = {
        'n_records': len(records),
        'total_i_beats': total_i_beats,
        'total_ii_beats': total_ii_beats,
        'total_i_plots': total_i_plots,
        'total_ii_plots': total_ii_plots,
        'hr_mean': round(sum(hrs) / len(hrs), 1) if hrs else None,
        'hr_range': (round(min(hrs), 1), round(max(hrs), 1)) if hrs else None,
        'p_axis_mean': round(sum(p_axes) / len(p_axes), 1) if p_axes else None,
        'qrs_axis_mean': round(sum(qrs_axes) / len(qrs_axes), 1) if qrs_axes else None,
    }

    return _write_html(output_dir, records, title, aggregates)


def _write_html(output_dir: str, records: list[dict], title: str,
                aggregates: dict | None = None) -> str:
    """Render the HTML page."""
    rows_html = ''
    for i, r in enumerate(records):
        hr_str = f'{r["hr"]:.0f}' if r['hr'] is not None else '—'
        p_axis_str = f'{r["p_axis"]:.0f}°' if r['p_axis'] is not None else '—'
        qrs_axis_str = f'{r["qrs_axis"]:.0f}°' if r['qrs_axis'] is not None else '—'
        interp = (r['interpretation'] or '')[:80]

        # Image thumbnails
        seg_i_img = (f'<a href="{r["name"]}/lead_I/segmentation.png" target="_blank">'
                     f'<img src="{r["name"]}/lead_I/segmentation.png" '
                     f'class="thumb" title="Lead I segmentation" loading="lazy"></a>'
                     if r['has_seg_i'] else '<span class="na">—</span>')
        seg_ii_img = (f'<a href="{r["name"]}/lead_II/segmentation.png" target="_blank">'
                      f'<img src="{r["name"]}/lead_II/segmentation.png" '
                      f'class="thumb" title="Lead II segmentation" loading="lazy"></a>'
                      if r['has_seg_ii'] else '<span class="na">—</span>')

        rows_html += f'''
        <tr>
            <td class="idx">{i + 1}</td>
            <td class="rec"><a href="{r["name"]}/">{r["name"]}</a></td>
            <td class="num">{r["i_beats"]}</td>
            <td class="num">{r["ii_beats"]}</td>
            <td class="num">{r["i_pwaves"]}</td>
            <td class="num">{r["ii_pwaves"]}</td>
            <td class="num">{hr_str}</td>
            <td class="num">{p_axis_str}</td>
            <td class="num">{qrs_axis_str}</td>
            <td class="interp" title="{interp}">{interp}</td>
            <td class="img">{seg_i_img}</td>
            <td class="img">{seg_ii_img}</td>
        </tr>'''

    # Stats bar
    stats_html = ''
    if aggregates:
        stats_html = f'''
        <div class="stats">
            <span>Records: <b>{aggregates["n_records"]}</b></span>
            <span>Lead I beats: <b>{aggregates["total_i_beats"]}</b></span>
            <span>Lead II beats: <b>{aggregates["total_ii_beats"]}</b></span>
            <span>Plots: <b>{aggregates["total_i_plots"]}</b> (I) / <b>{aggregates["total_ii_plots"]}</b> (II)</span>
            <span>HR: <b>{aggregates["hr_mean"]}</b> bpm ({aggregates["hr_range"][0]}–{aggregates["hr_range"][1]})</span>
            <span>P axis: <b>{aggregates["p_axis_mean"]}°</b></span>
            <span>QRS axis: <b>{aggregates["qrs_axis_mean"]}°</b></span>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #1a1a2e; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 24px 32px; }}
.header h1 {{ font-size: 22px; font-weight: 600; }}
.header p {{ font-size: 13px; opacity: 0.7; margin-top: 4px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 14px 32px; background: white; border-bottom: 1px solid #e0e0e0; font-size: 13px; }}
.stats span {{ white-space: nowrap; }}
.stats b {{ color: #2196f3; }}
.toolbar {{ padding: 12px 32px; background: white; display: flex; gap: 10px; align-items: center; }}
.toolbar input {{ padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; width: 220px; }}
.toolbar select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; background: white; table-layout: fixed; }}
thead {{ position: sticky; top: 0; z-index: 1; }}
thead th {{ background: #fafafa; border-bottom: 2px solid #e0e0e0; padding: 10px 8px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #666; text-align: left; cursor: pointer; user-select: none; }}
thead th:hover {{ color: #2196f3; }}
thead th.num {{ text-align: center; }}
tbody td {{ padding: 8px; border-bottom: 1px solid #f0f0f0; font-size: 13px; vertical-align: middle; }}
tbody tr:hover {{ background: #f5f8ff; }}
td.idx {{ color: #999; text-align: center; width: 36px; }}
td.rec {{ font-weight: 500; width: 130px; }}
td.rec a {{ color: #1565c0; text-decoration: none; }}
td.rec a:hover {{ text-decoration: underline; }}
td.num {{ text-align: center; width: 60px; }}
td.interp {{ font-size: 11px; color: #666; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
td.img {{ text-align: center; width: 80px; }}
.thumb {{ height: 48px; border-radius: 3px; border: 1px solid #e0e0e0; transition: transform 0.15s; }}
.thumb:hover {{ transform: scale(2.2); box-shadow: 0 4px 12px rgba(0,0,0,0.15); position: relative; z-index: 10; }}
.na {{ color: #ccc; font-size: 11px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.footer {{ padding: 16px 32px; text-align: center; font-size: 11px; color: #999; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>{title}</h1>
    <p>HSMM P-QRS-T waveform segmentation — Lead I + Lead II with refined boundaries</p>
</div>
{stats_html}
<div class="toolbar">
    <input type="text" id="search" placeholder="🔍  Filter by record name or interpretation..." oninput="filterTable()">
    <select id="sortBy" onchange="sortTable()">
        <option value="0">Sort by: #</option>
        <option value="1">Record name</option>
        <option value="2">Lead I beats</option>
        <option value="3">Lead II beats</option>
        <option value="6">Heart rate</option>
        <option value="7">P axis</option>
        <option value="8">QRS axis</option>
    </select>
    <span id="rowCount" style="font-size:12px;color:#999;margin-left:auto;"></span>
</div>
<table>
<thead>
<tr>
    <th class="num" onclick="sortByCol(0)">#</th>
    <th onclick="sortByCol(1)">Record</th>
    <th class="num" onclick="sortByCol(2)">I Beats</th>
    <th class="num" onclick="sortByCol(3)">II Beats</th>
    <th class="num" onclick="sortByCol(4)">I PWaves</th>
    <th class="num" onclick="sortByCol(5)">II PWaves</th>
    <th class="num" onclick="sortByCol(6)">HR</th>
    <th class="num" onclick="sortByCol(7)">P Axis</th>
    <th class="num" onclick="sortByCol(8)">QRS Axis</th>
    <th onclick="sortByCol(9)">Interpretation</th>
    <th class="num">Lead I Seg</th>
    <th class="num">Lead II Seg</th>
</tr>
</thead>
<tbody id="tableBody">
{rows_html}
</tbody>
</table>
<div class="footer">
    Generated from {os.path.abspath(output_dir)} &mdash; {len(records)} records
</div>
</div>

<script>
let sortCol = -1, sortDir = 1;

function filterTable() {{
    const q = document.getElementById('search').value.toLowerCase();
    const rows = document.querySelectorAll('#tableBody tr');
    let visible = 0;
    rows.forEach(row => {{
        const text = row.textContent.toLowerCase();
        const show = !q || text.includes(q);
        row.style.display = show ? '' : 'none';
        if (show) visible++;
    }});
    document.getElementById('rowCount').textContent = visible + ' / ' + rows.length + ' records';
}}

function sortByCol(col) {{
    if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
    document.getElementById('sortBy').value = String(col);
    doSort();
}}

function sortTable() {{
    sortCol = parseInt(document.getElementById('sortBy').value);
    sortDir = 1;
    doSort();
}}

function doSort() {{
    if (sortCol < 0) return;
    const tbody = document.getElementById('tableBody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
        let va = a.cells[sortCol].textContent.trim();
        let vb = b.cells[sortCol].textContent.trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return (na - nb) * sortDir;
        return va.localeCompare(vb) * sortDir;
    }});
    rows.forEach(row => tbody.appendChild(row));
    // Re-number
    rows.forEach((row, i) => {{ row.cells[0].textContent = i + 1; }});
}}

filterTable();
</script>
</body>
</html>'''

    out_path = os.path.join(output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return out_path


# ---- CLI entry point ----
if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else (
        'c:/LoyaltyLo/PythonProjects/ECG_engineering/ecg_waveform_extraction/output_rala_full/_p_qrs_t_wave'
    )
    path = build_dashboard(out)
    print(f'Dashboard generated: {path}')
