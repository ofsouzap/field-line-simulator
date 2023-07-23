from typing import TextIO, Tuple, Any, Optional
from os.path import isfile


__SETTINGS_DEFAULT_FILENAME = "settings.conf"


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


def set_default_settings() -> None:

    global show_field_line_arrows
    global field_line_count_factor
    global field_line_trace_step_distance_screen_space
    global field_line_trace_max_step_count
    global field_line_trace_element_stop_distance_screen_space
    global field_line_render_arrowhead_spacing
    global auto_recalcualate

    show_field_line_arrows = True
    field_line_count_factor = 4.0
    field_line_trace_step_distance_screen_space = 10
    field_line_trace_max_step_count = 500
    field_line_trace_element_stop_distance_screen_space = 1
    field_line_render_arrowhead_spacing = 100
    auto_recalcualate = True


def __write_setting(stream: TextIO, name: str, val):
    stream.write(f"{name}={str(val)}\n")


def __str_of_bool(b: bool) -> str:
    if b:
        return "True"
    else:
        return "False"


def save_settings(filename: str = __SETTINGS_DEFAULT_FILENAME) -> None:

    with open(filename, "w") as file:

        __write_setting(file, "show_field_line_arrows", __str_of_bool(show_field_line_arrows))
        __write_setting(file, "field_line_count_factor", field_line_count_factor)
        __write_setting(file, "field_line_trace_step_distance_screen_space", field_line_trace_step_distance_screen_space)
        __write_setting(file, "field_line_trace_max_step_count", field_line_trace_max_step_count)
        __write_setting(file, "field_line_trace_element_stop_distance_screen_space", field_line_trace_element_stop_distance_screen_space)
        __write_setting(file, "field_line_render_arrowhead_spacing", field_line_render_arrowhead_spacing)
        __write_setting(file, "auto_recalcualate", __str_of_bool(auto_recalcualate))


def __read_setting(stream: TextIO) -> Optional[Tuple[str, Any]]:

    line = stream.readline()

    if not line:
        return None

    line = line.strip()

    parts = line.split("=")

    assert len(parts) == 2, f"Invalid config line: {line}"

    return (parts[0], parts[1])


def __read_bool(s: str) -> bool:
    if s.lower() in ["y", "yes", "true", "1"]:
        return True
    elif s.lower() in ["n", "no", "false", "0"]:
        return False
    else:
        raise ValueError(s)


def load_settings(filename: str = __SETTINGS_DEFAULT_FILENAME) -> None:

    global show_field_line_arrows
    global field_line_count_factor
    global field_line_trace_step_distance_screen_space
    global field_line_trace_max_step_count
    global field_line_trace_element_stop_distance_screen_space
    global field_line_render_arrowhead_spacing
    global auto_recalcualate

    if not isfile(filename):

        set_default_settings()

    else:

        with open(filename, "r") as file:

            while True:

                out = __read_setting(file)

                if not out:

                    break

                else:

                    name, val = out

                    if name == "show_field_line_arrows":
                        show_field_line_arrows = __read_bool(val)
                    elif name == "field_line_count_factor":
                        field_line_count_factor = float(val)
                    elif name == "field_line_trace_step_distance_screen_space":
                        field_line_trace_step_distance_screen_space = float(val)
                    elif name == "field_line_trace_max_step_count":
                        field_line_trace_max_step_count = int(val)
                    elif name == "field_line_trace_element_stop_distance_screen_space":
                        field_line_trace_element_stop_distance_screen_space = float(val)
                    elif name == "field_line_render_arrowhead_spacing":
                        field_line_render_arrowhead_spacing = int(val)
                    elif name == "auto_recalcualate":
                        auto_recalcualate = __read_bool(val)
