import cProfile as prof
from pstats import Stats, SortKey
import numpy as np
from field import Field
from point_source import PointSource

OUT_STATS_FILENAME = "field_line_tracing.pstats"

EPS = 1e-6

def trace_lines(field: Field,
                line_starts: np.ndarray,
                max_points: int,
                positives: np.ndarray,
                clip_ranges: np.ndarray) -> None:

    field.trace_field_lines(
        starts=line_starts,
        max_points=max_points,
        positives=positives,
        clip_ranges=clip_ranges
    )

max_points = 10000

# (pos_x, pos_y, strength)
sources = [
    (20, 20, 0.05),
    (40, 20, -0.05),
    (30, 20, 0.01),
    (30, 30, -0.01),
    (1, 0, 0.1)
]

line_starts_list = []
positives_list = []
for s in sources:

    phi = np.linspace(0, 2*np.pi, 24, endpoint=False)
    dx = np.cos(phi) * EPS
    dy = np.sin(phi) * EPS

    for i in range(dx.shape[0]):
        line_starts_list.append([s[0]+dx[i], s[1]+dy[i]])
        positives_list.append(s[2] > 0)

line_starts = np.array(line_starts_list)
positives = np.array(positives_list)

clip_ranges = np.array([
    [-50, 50],
    [-50, 50]
])

field = Field()

for s in sources:
    field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

prof.run("trace_lines(field, line_starts, max_points, positives, clip_ranges)", OUT_STATS_FILENAME)

stats = Stats(OUT_STATS_FILENAME)
stats \
    .strip_dirs() \
    .sort_stats(SortKey.TIME) \
    .print_stats()
