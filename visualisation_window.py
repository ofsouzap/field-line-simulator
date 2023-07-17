import pyglet
from field import Field, List
from field_element import PointSource
import numpy as np
from _debug_util import Timer


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


class AppAlreadyRunningException(Exception): pass


class Window(pyglet.window.Window):

    def __init__(self, width: int, height: int):

        super().__init__(width, height, WINDOW_TITLE)

        self.batch = pyglet.graphics.Batch()

        self.__lifetime = 0
        self.__field_line_lines = []

    def draw_field(self,
                   field: Field) -> None:
        """Clears the current diagram and draws the field provided"""

        # Clear current lines

        self.__clear_field_lines()

        # Generate field line generation data

        clip_ranges = np.array([
            [0, self.width],
            [0, self.height]
        ])

        line_starts_list = []
        postives_list = []

        for ele in field.iter_elements():

            starts, pos = ele.get_field_line_starts(fac=16)
            line_starts_list.append(starts)
            postives_list.append(pos)

        line_starts = np.concatenate(line_starts_list)
        positives = np.concatenate(postives_list)

        # Generate field line data

        with Timer("Trace Lines"):  # TODO - remove timers when ready
            field_lines = field.trace_field_lines(
                line_starts,
                500,
                positives,
                clip_ranges=clip_ranges
            )

        # Plot calculated lines

        with Timer("Plot Lines"):  # TODO - remove timers when ready
            self.__add_field_lines(field_lines)

    def __add_field_lines(self,
                          lines: np.ndarray) -> None:
        for points in lines:
            self.__add_field_line(points)

    def __add_field_line(self,
                         points: np.ndarray) -> None:

        # If not enough points then don't draw anything

        if points.shape[0] <= 1:
            return

        # Draw parts of line

        prev = points[0]

        for curr in points[1:]:

            # If point is repeated (probably meaning the line was clipped) then stop drawing
            if np.all(np.isclose(prev, curr)):
                break 

            line = pyglet.shapes.Line(
                prev[0], prev[1],
                curr[0], curr[1],
                batch=self.batch
            )

            self.__field_line_lines.append(line)

            prev = curr

    def __clear_field_lines(self) -> None:

        self.__field_line_lines.clear()

    def round_float_pos(self, pos: np.ndarray) -> np.ndarray:

        return np.around(pos, decimals=0).astype(int)

    def on_draw(self) -> None:
        """Clear screen and draw batch of shapes"""

        self.clear()
        self.batch.draw()

    def update(self, delta_time: float) -> None:

        self.__lifetime += delta_time


class Controller:
    """A wrapper for a visualisation window for controlling the window"""

    app_running: bool = False

    def __init__(self, window: Window):

        self.__window = window

    def set_field(self, field: Field) -> None:
        self.__window.draw_field(field)

    @staticmethod
    def run_app() -> None:
        """Starts the Pyglet event loop running.
Only should be called when all windows have been created and can't be called again from any instance of the class or otherwise.
Will block until all windows are closed.
"""

        if Controller.app_running:

            raise AppAlreadyRunningException()

        else:

            pyglet.app.run()


def create_window() -> Controller:
    """Create and open the visualisation window.
Note that this doesn't start the window running

Returns:

    controller - a controller object for the window
"""

    window = Window(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)  # Create main window
    controller = Controller(window)  # Create window's controller

    # TODO - below field-making is just an example, delete once proper gui with field editing is made

    field = Field()

    sources_dat = [
        (200, 200, 5),
        (400, 200, -5),
        # (300, 200, 1),
        # (300, 300, -1),
        # (10, 0, 10)
    ]

    source_shapes: List[pyglet.shapes.ShapeBase] = []

    for i, s in enumerate(sources_dat):

        source_shapes.append(pyglet.shapes.Circle(s[0], s[1], 3, batch=window.batch))

        field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

    window.draw_field(field)

    # Schedule window's update function

    pyglet.clock.schedule_interval(window.update, 1/30)

    # Return the window controller

    return controller


if __name__ == "__main__":

    controller = create_window()
    controller.run_app()
