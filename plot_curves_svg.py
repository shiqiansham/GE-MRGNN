import sys
from pathlib import Path

def read_csv(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    data = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        parts = line.strip().split(',')
        if len(parts) != 3:
            continue
        try:
            t = float(parts[0])
            a = float(parts[1])
            b = float(parts[2])
        except:
            continue
        data.append((t, a, b))
    return data

def polyline(points, xscale, yscale, xoff, yoff, color):
    if not points:
        return ''
    d = []
    for x, y in points:
        px = xoff + x * xscale
        py = yoff - y * yscale
        d.append(f"{px:.2f},{py:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(d)}"/>'

def axes(width, height, margin):
    x0 = margin
    y0 = height - margin
    x1 = width - margin
    y1 = margin
    return f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>'

def write_svg(path, title, series, width=640, height=420, margin=40):
    xoff = margin
    yoff = height - margin
    xscale = (width - 2 * margin)
    yscale = (height - 2 * margin)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.append(f'<text x="{width/2:.2f}" y="24" text-anchor="middle" font-size="16" fill="#222">{title}</text>')
    svg.append(axes(width, height, margin))
    for color, pts in series:
        svg.append(polyline(pts, xscale, yscale, xoff, yoff, color))
    svg.append(f'<text x="{width/2:.2f}" y="{height-8}" text-anchor="middle" font-size="12" fill="#555">X</text>')
    svg.append(f'<text x="12" y="{height/2:.2f}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90 12,{height/2:.2f})">Y</text>')
    svg.append('</svg>')
    Path(path).write_text('\n'.join(svg), encoding='utf-8')

def main():
    if '--roc' in sys.argv and '--out_dir' in sys.argv:
        roc_csv = Path(sys.argv[sys.argv.index('--roc') + 1])
        out_dir = Path(sys.argv[sys.argv.index('--out_dir') + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        roc_data = read_csv(roc_csv)
        roc_pts = [(fpr, tpr) for _, fpr, tpr in roc_data]
        write_svg(out_dir / 'roc.svg', 'ROC Curve', [('#2c7be5', roc_pts)])
    if '--pr' in sys.argv and '--out_dir' in sys.argv:
        pr_csv = Path(sys.argv[sys.argv.index('--pr') + 1])
        out_dir = Path(sys.argv[sys.argv.index('--out_dir') + 1])
        pr_data = read_csv(pr_csv)
        pr_pts = [(rec, prec) for _, prec, rec in pr_data]
        write_svg(out_dir / 'pr.svg', 'PR Curve', [('#00c26d', pr_pts)])

if __name__ == '__main__':
    main()
