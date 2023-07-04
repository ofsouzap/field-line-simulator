import pyglet
from field import Field, List
from point_source import PointSource
import numpy as np
from _debug_util import Timer


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


class DiagramWindow(pyglet.window.Window):

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


def open_gui():
    """Create and open the main GUI. Blocks until the window is closed"""

    window = DiagramWindow(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)  # Create main window

    # TODO - below field-making is just an example, delete once proper gui with field editing is made

    field = Field()

    sources_dat = [
        (200, 200, 5),
        (400, 200, -5),
        (300, 200, 1),
        (300, 300, -1),
        (10, 0, 10)
    ]

    source_shapes: List[pyglet.shapes.ShapeBase] = []

    for i, s in enumerate(sources_dat):

        source_shapes.append(pyglet.shapes.Circle(s[0], s[1], 3, batch=window.batch))

        field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

    window.draw_field(field)

    pyglet.clock.schedule_interval(window.update, 1/30)  # Register window's update function

    pyglet.app.run()  # Run app


if __name__ == "__main__":
    open_gui()
