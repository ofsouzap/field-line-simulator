EPS = 1e-6
VIEWPORT_SCALE_FAC: float = 10
"""The number of units of distance in the field per pixel of display"""

field_line_trace_step_distance: int = 10 * VIEWPORT_SCALE_FAC
field_line_trace_element_stop_distance: int = 1 * VIEWPORT_SCALE_FAC

field_line_render_arrowhead_spacing: int = 100
"""The spacing in screen space between arrowheads drawn on field lines"""

auto_recalcualate: bool = True