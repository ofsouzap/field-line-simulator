from numpy import array as np_array

EPS = 1e-6
VIEWPORT_SCALE_FAC: float = 10
"""The number of units of distance in the field per pixel of display"""

field_line_trace_step_distance: int = 5 * VIEWPORT_SCALE_FAC
field_line_trace_element_stop_distance: int = 1 * VIEWPORT_SCALE_FAC

auto_recalcualate: bool = True