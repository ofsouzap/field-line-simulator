import pyglet
import numpy as np
from field import Field
from point_source import PointSource
from vectors import EPS


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


class MainWindow(pyglet.window.Window):

    def __init__(self, width: int, height: int):

        super().__init__(width, height, WINDOW_TITLE)

        self.batch = pyglet.graphics.Batch()

        self.__field_line_lines = []

        # TODO: Below is just for trying stuff out. Remove later

        field = Field()

        clip_ranges = np.array([
            [0, width],
            [0, height]
        ])

        sources = [
            (200, 200, 5),
            (400, 200, -5),
            (300, 200, 1),
            (300, 300, -1),
            (0, 0, 100)
        ]

        for i, s in enumerate(sources):

            self.__dict__["source_"+str(i)] = pyglet.shapes.Circle(s[0], s[1], 3, batch=self.batch)

            field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

        phi = np.linspace(0, 2*np.pi, 24, endpoint=False)
        dx = np.cos(phi) * EPS
        dy = np.sin(phi) * EPS

        # dx = np.array([[200.1, 0.0]])
        # dy = np.array([[0.0, 0.0]])

        for source in sources:

            line_starts = np.dstack((dx+source[0], dy+source[1]))[0]
            positives = np.repeat(source[2] > 0, line_starts.shape[0])

            field_lines = field.trace_field_lines(
                line_starts,
                500,
                positives,
                clip_ranges=clip_ranges
            )

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

        pass


def main():

    window = MainWindow(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)  # Create main window

    pyglet.clock.schedule_interval(window.update, 1/30)  # Register window's update function

    pyglet.app.run()  # Run app


if __name__ == "__main__":
    main()