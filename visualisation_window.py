import pyglet
from abc import ABC, abstractmethod
from typing import Optional, Callable, Set, Tuple
import vectors
from field import Field
from field_element import ElementBase, PointSource
import numpy as np
from _debug_util import Timer


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


class AppAlreadyRunningException(Exception): pass


class _FieldElementRenderBase(ABC):

    @abstractmethod
    def draw(self, batch: pyglet.graphics.Batch) -> Set:
        """Creates the shapes required for rendering the element, adds them to the batch and then returns a set of the shapes created.

Parameters:

    batch - the batch to add the shapes to

Returns:

    shapes - the shapes created and added to the batch. These should be stored so they aren't garbage collected
"""
        raise NotImplementedError()

    @abstractmethod
    def point_in_bounds(self, posx: int, posy: int) -> bool:
        """Checks if the point specified lies in the region that the rendered element is in"""
        raise NotImplementedError()


class _PointSourceRender(_FieldElementRenderBase):

    RADIUS: int = 5
    SQR_RADIUS: int = RADIUS * RADIUS

    def __init__(self, ps: PointSource):
        self.ps = ps

    def get_color(self) -> Tuple[int, int, int, int]:

        if self.ps.strength == 0:
            return WHITE
        elif self.ps.strength > 0:
            return (
                255,
                128*(1-round(np.tanh(self.ps.strength/50))),
                128*(1-round(np.tanh(self.ps.strength/50))),
                255
            )
        else:
            return (
                128*(1-round(np.tanh(-self.ps.strength/50))),
                128*(1-round(np.tanh(-self.ps.strength/50))),
                255,
                255
            )

    def draw(self, batch: pyglet.graphics.Batch) -> Set:

        circle = pyglet.shapes.Circle(
            x=round(self.ps.x),
            y=round(self.ps.y),
            radius=_PointSourceRender.RADIUS,
            color=self.get_color(),
            batch=batch
        )

        return {circle}

    def point_in_bounds(self, posx: int, posy: int) -> bool:

        pos_vec = np.array([posx, posy], dtype=float)

        sqr_dist = vectors.sqr_magnitudes(pos_vec - self.ps.pos)[0]

        return sqr_dist <= _PointSourceRender.SQR_RADIUS


def _create_element_renderer(ele: ElementBase) -> _FieldElementRenderBase:

    match ele:

        case PointSource():
            return _PointSourceRender(ele)

        case _:
            raise ValueError("Unhandled element class")


class Window(pyglet.window.Window):

    def __init__(self,
                 width: int,
                 height: int,
                 on_mouse_press: Callable[[int, int, int, int], None],):

        super().__init__(width, height, WINDOW_TITLE)

        self.__lifetime = 0

        self.field_lines_batch = pyglet.graphics.Batch()
        self.field_elements_batch = pyglet.graphics.Batch()
        self.__field_shapes: Set = set()

        self.mouse_press_callback = on_mouse_press

    def draw_field_elements(self,
                            field: Field) -> None:
        """Draws the field elements of a field without drawing the field lines"""

        for ele in field.iter_elements():

            shapes: Set = _create_element_renderer(ele).draw(self.field_elements_batch)

            self.__field_shapes |= shapes

    def draw_field_lines(self,
                         field: Field) -> None:
        """Draws the field lines of a field without drawing the field elements"""

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

        if len(line_starts_list) == 0:
            return

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
                batch=self.field_lines_batch
            )

            self.__field_shapes.add(line)

            prev = curr

    def clear_screen(self) -> None:

        self.__field_shapes.clear()

    def round_float_pos(self, pos: np.ndarray) -> np.ndarray:

        return np.around(pos, decimals=0).astype(int)

    def on_draw(self) -> None:
        """Clear screen and draw batch of shapes"""

        self.clear()

        # Draw field lines and then field elements so that field elements are still visible

        self.field_lines_batch.draw()
        self.field_elements_batch.draw()

    def update(self, delta_time: float) -> None:

        self.__lifetime += delta_time

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_press_callback(x, y, button, modifiers)


class Controller:
    """A wrapper for a visualisation window for controlling the window"""

    app_running: bool = False

    def __init__(self,
                 window: Window,
                 field: Optional[Field] = None):

        self.__window = window

        if field is not None:
            self.__field = field
        else:
            self.__field = Field()

    def recalculate(self) -> None:

        self.redraw_only_elements()
        self.__window.draw_field_lines(self.__field)

    def redraw_only_elements(self) -> None:

        self.__window.clear_screen()
        self.__window.draw_field_elements(self.__field)

    def add_field_element(self, ele: ElementBase) -> None:

        self.__field.add_element(ele)

        self.redraw_only_elements()

    def try_delete_field_element_at(self, posx: int, posy: int) -> bool:
        """Tries to remove an element at the position specified. Returns whether one was found"""

        for ele in self.__field.iter_elements():

            if _create_element_renderer(ele).point_in_bounds(posx, posy):
                self.__field.remove_element(ele)
                self.redraw_only_elements()
                return True

        return False

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


def create_window(
        on_mouse_press: Callable[[int, int, int, int], None]
    ) -> Controller:
    """Create and open the visualisation window.
Note that this doesn't start the window running

Parameters:

    on_mouse_press - a callable run when the window is clicked on with the mouse. \
The inputs to the callable are the same as those for pyglet.Window.on_mouse_press (x, y, button, modifiers)

Returns:

    controller - a controller object for the window
"""

    window = Window(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT, on_mouse_press)  # Create main window
    controller = Controller(window)  # Create window's controller

    # Schedule window's update function

    pyglet.clock.schedule_interval(window.update, 1/30)

    # Return the window controller

    return controller


if __name__ == "__main__":

    controller = create_window(lambda x, y, btn, mods: print(f"Click at {x}, {y} with button {btn}"))
    controller.run_app()
