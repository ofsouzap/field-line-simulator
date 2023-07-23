EPS = 1e-6
VIEWPORT_SCALE_FAC: float = 10
"""The number of units of distance in the field per pixel of display"""

show_field_line_arrows: bool = True
field_line_count_factor: float = 4.0
field_line_trace_step_distance_screen_space: float = 10
"""How far to step the field lines each step (in screen space)"""
field_line_trace_max_step_count: int = 500
field_line_trace_element_stop_distance_screen_space: float = 1
"""How far from a field element to stop a field line (in screen space)"""

field_line_render_arrowhead_spacing: int = 100
"""The spacing in screen space between arrowheads drawn on field lines"""

auto_recalcualate: bool = True