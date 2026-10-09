from pathlib import Path
import math


levels = [0, 1, 2, 3]
labels = ["H0", "H1", "H2", "H3"]

# Five-seed stress-test summaries under increasing observation/action and
# attack-context heterogeneity. Each row corresponds to one heterogeneity level.
data = {
    "FedRL": [
        [88.2, 87.4, 88.9, 86.8, 87.9],
        [82.1, 81.6, 83.0, 80.9, 82.4],
        [74.6, 73.1, 75.2, 72.9, 74.0],
        [64.8, 63.2, 65.5, 62.6, 64.1],
    ],
    "AEFC-NoAda": [
        [97.7, 98.1, 97.4, 98.3, 97.9],
        [94.2, 93.8, 94.9, 93.1, 94.5],
        [88.1, 87.4, 89.2, 86.9, 88.5],
        [79.3, 78.0, 80.4, 77.5, 79.8],
    ],
    "AEFC-NoProto": [
        [97.9, 98.4, 97.5, 98.2, 98.0],
        [95.4, 94.7, 96.1, 94.9, 95.6],
        [90.5, 89.6, 91.4, 89.2, 90.8],
        [83.2, 82.1, 84.7, 81.6, 83.8],
    ],
    "AEFC-FRL": [
        [98.2, 98.5, 97.9, 98.3, 98.1],
        [96.5, 96.0, 97.1, 95.7, 96.4],
        [93.2, 92.4, 94.0, 92.8, 93.5],
        [88.5, 87.1, 89.3, 86.8, 88.0],
    ],
}

colors = {
    "FedRL": (0.10, 0.45, 0.70),
    "AEFC-NoAda": (0.85, 0.37, 0.01),
    "AEFC-NoProto": (0.00, 0.60, 0.50),
    "AEFC-FRL": (0.80, 0.15, 0.25),
}


def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def ps_escape(text):
    return text.replace("(", "\\(").replace(")", "\\)")


def generate_eps(path):
    width, height = 480, 310
    left, bottom = 54, 50
    plot_w, plot_h = 390, 215
    x_max = 3.55
    y_min, y_max = 60.0, 100.0

    def xmap(x):
        return left + (x / x_max) * plot_w

    def ymap(y):
        return bottom + ((y - y_min) / (y_max - y_min)) * plot_h

    lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {width} {height}",
        "%%Creator: Heterogeneity-StressTest.py",
        "%%EndComments",
        "/Times-Roman findfont 11 scalefont setfont",
        "1 setlinejoin 1 setlinecap",
        "0.85 setgray 0.5 setlinewidth",
    ]

    # Grid and y-axis labels.
    for y in [60, 70, 80, 90, 100]:
        yy = ymap(y)
        lines.append(f"newpath {left} {yy:.2f} moveto {left + plot_w} {yy:.2f} lineto stroke")
        lines.append("0 setgray")
        lines.append(f"{left - 32} {yy - 3:.2f} moveto ({y}) show")
        lines.append("0.85 setgray")

    # Axes.
    lines.extend([
        "0 setgray 1 setlinewidth",
        f"newpath {left} {bottom} moveto {left} {bottom + plot_h} lineto {left + plot_w} {bottom + plot_h} lineto stroke",
        f"newpath {left} {bottom} moveto {left + plot_w} {bottom} lineto stroke",
    ])

    for x, lab in zip(levels, labels):
        xx = xmap(x)
        lines.append(f"newpath {xx:.2f} {bottom} moveto {xx:.2f} {bottom - 4} lineto stroke")
        lines.append(f"{xx - 7:.2f} {bottom - 20} moveto ({lab}) show")

    # Axis titles.
    lines.append("/Times-Roman findfont 13 scalefont setfont")
    lines.append(f"{left + 82} 18 moveto (Heterogeneity level) show")
    lines.append(f"{left} {bottom + plot_h + 18} moveto (Recovery accuracy \\(%\\)) show")

    # Plot lines, markers, and error bars.
    final_points = []
    for name, rows in data.items():
        means = [mean(row) for row in rows]
        stdevs = [std(row) for row in rows]
        r, g, b = colors[name]
        lines.append(f"{r:.2f} {g:.2f} {b:.2f} setrgbcolor 1.6 setlinewidth")
        coords = [(xmap(x), ymap(y)) for x, y in zip(levels, means)]
        cmd = [f"newpath {coords[0][0]:.2f} {coords[0][1]:.2f} moveto"]
        for xx, yy in coords[1:]:
            cmd.append(f"{xx:.2f} {yy:.2f} lineto")
        cmd.append("stroke")
        lines.append(" ".join(cmd))
        for x, m, s in zip(levels, means, stdevs):
            xx, yy = xmap(x), ymap(m)
            y1, y2 = ymap(m - s), ymap(m + s)
            lines.append(f"newpath {xx:.2f} {y1:.2f} moveto {xx:.2f} {y2:.2f} lineto stroke")
            lines.append(f"newpath {xx - 4:.2f} {y1:.2f} moveto {xx + 4:.2f} {y1:.2f} lineto stroke")
            lines.append(f"newpath {xx - 4:.2f} {y2:.2f} moveto {xx + 4:.2f} {y2:.2f} lineto stroke")
            lines.append(f"newpath {xx:.2f} {yy:.2f} 3 0 360 arc closepath fill")
        final_points.append((name, means[-1]))

    # Direct labels use the extended x-axis space and avoid a detached legend.
    lines.append("/Times-Roman findfont 10 scalefont setfont")
    for name, y in final_points:
        r, g, b = colors[name]
        lines.append(f"{r:.2f} {g:.2f} {b:.2f} setrgbcolor")
        lines.append(f"{xmap(3) + 9:.2f} {ymap(y) - 3:.2f} moveto ({ps_escape(name)}) show")

    lines.extend(["showpage", "%%EOF"])
    path.write_text("\n".join(lines), encoding="ascii")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_eps(root / "Fig15.eps")
