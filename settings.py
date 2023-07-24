from typing import TextIO, Tuple, Any, Optional
from os.path import isfile


_SETTINGS_DEFAULT_FILENAME = "settings.conf"


class Settings:

    EPS = 1e-6
    VIEWPORT_SCALE_FAC: float = 10
    """The number of units of distance in the field per pixel of display"""

    def __init__(self):

        self.show_field_line_arrows: bool = True
        self.field_line_count_factor: float = 4.0
        self.field_line_trace_step_distance_screen_space: float = 10
        """How far to step the field lines each step (in screen space)"""
        self.field_line_trace_max_step_count: int = 500
        self.field_line_trace_element_stop_distance_screen_space: float = 1
        """How far from a field element to stop a field line (in screen space)"""

        self.field_line_render_arrowhead_spacing: int = 100
        """The spacing in screen space between arrowheads drawn on field lines"""

        self.auto_recalcualate: bool = True

    def set_default_settings(self) -> None:

        self.show_field_line_arrows = True
        self.field_line_count_factor = 4.0
        self.field_line_trace_step_distance_screen_space = 10
        self.field_line_trace_max_step_count = 500
        self.field_line_trace_element_stop_distance_screen_space = 1
        self.field_line_render_arrowhead_spacing = 100
        self.auto_recalcualate = True

    def __write_setting(self, stream: TextIO, name: str, val):
        stream.write(f"{name}={str(val)}\n")

    @staticmethod
    def __str_of_bool(b: bool) -> str:
        if b:
            return "True"
        else:
            return "False"

    def save_settings(self, filename: str = _SETTINGS_DEFAULT_FILENAME) -> None:

        with open(filename, "w") as file:

            self.__write_setting(file, "show_field_line_arrows", self.__str_of_bool(self.show_field_line_arrows))
            self.__write_setting(file, "field_line_count_factor", self.field_line_count_factor)
            self.__write_setting(file, "field_line_trace_step_distance_screen_space", self.field_line_trace_step_distance_screen_space)
            self.__write_setting(file, "field_line_trace_max_step_count", self.field_line_trace_max_step_count)
            self.__write_setting(file, "field_line_trace_element_stop_distance_screen_space", self.field_line_trace_element_stop_distance_screen_space)
            self.__write_setting(file, "field_line_render_arrowhead_spacing", self.field_line_render_arrowhead_spacing)
            self.__write_setting(file, "auto_recalcualate", self.__str_of_bool(self.auto_recalcualate))


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


def load_settings(filename: str = _SETTINGS_DEFAULT_FILENAME) -> None:

    global settings

    if not isfile(filename):

        settings.set_default_settings()

    else:

        with open(filename, "r") as file:

            while True:

                out = __read_setting(file)

                if not out:

                    break

                else:

                    name, val = out

                    if name == "show_field_line_arrows":
                        settings.show_field_line_arrows = __read_bool(val)
                    elif name == "field_line_count_factor":
                        settings.field_line_count_factor = float(val)
                    elif name == "field_line_trace_step_distance_screen_space":
                        settings.field_line_trace_step_distance_screen_space = float(val)
                    elif name == "field_line_trace_max_step_count":
                        settings.field_line_trace_max_step_count = int(val)
                    elif name == "field_line_trace_element_stop_distance_screen_space":
                        settings.field_line_trace_element_stop_distance_screen_space = float(val)
                    elif name == "field_line_render_arrowhead_spacing":
                        settings.field_line_render_arrowhead_spacing = int(val)
                    elif name == "auto_recalcualate":
                        settings.auto_recalcualate = __read_bool(val)


settings = Settings()
