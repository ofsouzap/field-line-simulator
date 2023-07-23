import pyglet
from abc import ABC, abstractmethod
from typing import Optional, Callable, Set, Tuple
from os.path import join as joinpath
import vectors
from field import Field
from field_element import ElementBase, PointSource, ChargePlane
import settings
from shortcuts import RawCommand as KeyPressCommand
from shortcuts import MOD_SHIFT, MOD_CTRL, MOD_ALT
import numpy as np
from _debug_util import Timer


def _resource(*path: str) -> str:
    return joinpath("resources", *path)


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


ARROWHEAD_LENGTH: int = 5
STATUS_ICON_POSITION: Tuple[int, int] = (0, 0)


STATUS_ICON_RES_PATH_ADD = _resource("status_icons", "add.png")
STATUS_ICON_RES_PATH_DELETE = _resource("status_icons", "delete.png")
STATUS_ICON_RES_PATH_LOADING = _resource("status_icons", "loading.png")


status_icon_add = pyglet.image.load(STATUS_ICON_RES_PATH_ADD)
status_icon_delete = pyglet.image.load(STATUS_ICON_RES_PATH_DELETE)
status_icon_loading = pyglet.image.load(STATUS_ICON_RES_PATH_LOADING)


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


class AppAlreadyRunningException(Exception): pass
class AppNotRunningException(Exception): pass


class _FieldElementRenderBase(ABC):

    @abstractmethod
    def draw(self, draw_bounds: np.ndarray, batch: pyglet.graphics.Batch) -> Set:
        """Creates the shapes required for rendering the element, adds them to the batch and then returns a set of the shapes created.

Parameters:

    draw_bounds - the boundaries of the drawing area. Useful for drawing infinite elements

    batch - the batch to add the shapes to

Returns:

    shapes - the shapes created and added to the batch. These should be stored so they aren't garbage collected
"""
        raise NotImplementedError()

    @abstractmethod
    def point_in_draw_bounds(self, posx: int, posy: int) -> bool:
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

    def draw(self, draw_bounds: np.ndarray, batch: pyglet.graphics.Batch) -> Set:

        circle = pyglet.shapes.Circle(
            x=round(self.ps.x/settings.VIEWPORT_SCALE_FAC),
            y=round(self.ps.y/settings.VIEWPORT_SCALE_FAC),
            radius=_PointSourceRender.RADIUS,
            color=self.get_color(),
            batch=batch
        )

        return {circle}

    def point_in_draw_bounds(self, posx: int, posy: int) -> bool:

        pos_vec = np.array([posx, posy], dtype=float)

        sqr_dist = vectors.sqr_magnitudes(pos_vec - (self.ps.pos/settings.VIEWPORT_SCALE_FAC))[0]

        return sqr_dist <= _PointSourceRender.SQR_RADIUS


class _ChargePlaneRender(_FieldElementRenderBase):

    WIDTH: int = 5
    SQR_WIDTH: int = WIDTH * WIDTH

    def __init__(self, cp: ChargePlane):
        self.cp = cp

    def get_color(self) -> Tuple[int, int, int, int]:

        if self.cp.strength_density == 0:
            return WHITE
        elif self.cp.strength_density > 0:
            return (
                255,
                128*(1-round(np.tanh(self.cp.strength_density/50))),
                128*(1-round(np.tanh(self.cp.strength_density/50))),
                255
            )
        else:
            return (
                128*(1-round(np.tanh(-self.cp.strength_density/50))),
                128*(1-round(np.tanh(-self.cp.strength_density/50))),
                255,
                255
            )

    def draw(self, draw_bounds: np.ndarray, batch: pyglet.graphics.Batch) -> Set:

        xstart: float
        xend: float
        ystart: float
        yend: float

        if np.isclose(self.cp.normal[0], 0):

            # Plane is horizontal

            xstart = draw_bounds[0][0]
            xend = draw_bounds[0][1]

            ystart = self.cp.pos[1]
            yend = self.cp.pos[1]

        elif np.isclose(self.cp.normal[1], 0):

            # Plane is vertical

            xstart = self.cp.pos[0]
            xend = self.cp.pos[0]

            ystart = draw_bounds[1][0]
            yend = draw_bounds[1][1]

        else:

            plane_grad = -self.cp.normal[0] / self.cp.normal[1]

            xstart = draw_bounds[0][0]
            xend = draw_bounds[0][1]

            ystart = self.cp.pos[1] + ((xstart - self.cp.pos[0]) * plane_grad)
            yend = self.cp.pos[1] + ((xend - self.cp.pos[0]) * plane_grad)

        line = pyglet.shapes.Line(
            x=xstart/settings.VIEWPORT_SCALE_FAC,
            y=ystart/settings.VIEWPORT_SCALE_FAC,
            x2=xend/settings.VIEWPORT_SCALE_FAC,
            y2=yend/settings.VIEWPORT_SCALE_FAC,
            width=_ChargePlaneRender.WIDTH,
            color=self.get_color(),
            batch=batch
        )

        return {line}

    def point_in_draw_bounds(self, posx: int, posy: int) -> bool:

        pos_vec_screen = np.array([posx, posy])
        pos_vec_world = pos_vec_screen*settings.VIEWPORT_SCALE_FAC

        closest_plane_pos_world = vectors.plane_closest_point_to_point(
            self.cp.pos[np.newaxis, :],
            self.cp.normal[np.newaxis, :],
            pos_vec_world[np.newaxis, :]
        )[0]

        closest_plane_pos_screen = closest_plane_pos_world / settings.VIEWPORT_SCALE_FAC

        sqr_dist = vectors.sqr_magnitudes(closest_plane_pos_screen - pos_vec_screen)[0]

        return sqr_dist <= _ChargePlaneRender.SQR_WIDTH


def _create_element_renderer(ele: ElementBase) -> _FieldElementRenderBase:

    match ele:

        case PointSource():
            return _PointSourceRender(ele)

        case ChargePlane():
            return _ChargePlaneRender(ele)

        case _:
            raise ValueError("Unhandled element class")


class Window(pyglet.window.Window):

    def __init__(self,
                 width: int,
                 height: int,
                 on_exit: Callable[[], None],
                 on_mouse_press: Callable[[int, int, int, int], None],
                 on_key_press: Callable[[KeyPressCommand], None]):

        super().__init__(width, height, WINDOW_TITLE)

        self.__lifetime = 0

        self.__on_exit = on_exit

        self.field_lines_batch = pyglet.graphics.Batch()
        self.__field_lines_shapes: Set = set()
        self.field_elements_batch = pyglet.graphics.Batch()
        self.__field_elements_shapes: Set = set()

        self.add_mode_sprite = pyglet.sprite.Sprite(status_icon_add, x=STATUS_ICON_POSITION[0], y=STATUS_ICON_POSITION[1])
        self.delete_mode_sprite = pyglet.sprite.Sprite(status_icon_delete, x=STATUS_ICON_POSITION[0], y=STATUS_ICON_POSITION[1])
        self.__click_mode_sprite: Optional[pyglet.sprite.Sprite] = None

        self.mouse_press_callback = on_mouse_press
        self.key_press_callback = on_key_press

    def on_close(self):
        super().on_close()
        self.__on_exit()

    @property
    def clip_bounds(self) -> np.ndarray:
        """The range of positions in world-space that should be rendered"""
        return np.array([
            [0.0, self.width*settings.VIEWPORT_SCALE_FAC],
            [0.0, self.height*settings.VIEWPORT_SCALE_FAC]
        ])

    def set_click_mode_none(self) -> None:
        self.__click_mode_sprite = None

    def set_click_mode_add(self) -> None:
        self.__click_mode_sprite = self.add_mode_sprite

    def set_click_mode_delete(self) -> None:
        self.__click_mode_sprite = self.delete_mode_sprite

    def __clear_field_element_shapes(self) -> None:

        for shape in self.__field_elements_shapes:
            shape.delete()

        self.__field_elements_shapes.clear()

    def draw_field_elements(self,
                            field: Field) -> None:
        """Draws the field elements of a field without drawing the field lines"""

        self.__clear_field_element_shapes()

        for ele in field.iter_elements():

            self.switch_to()
            shapes: Set = _create_element_renderer(ele).draw(self.clip_bounds, self.field_elements_batch)

            self.__field_elements_shapes |= shapes

    def __clear_field_lines_shapes(self) -> None:

        for shape in self.__field_lines_shapes:
            shape.delete()

        self.__field_lines_shapes.clear()

    def draw_field_lines(self,
                         field: Field) -> None:
        """Draws the field lines of a field without drawing the field elements"""

        self.__clear_field_lines_shapes()

        # Generate field line generation data

        line_starts_list = []
        postives_list = []

        for ele in field.iter_elements():

            starts, pos = ele.get_field_line_starts(self.clip_bounds, fac=settings.field_line_count_factor)
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
                settings.field_line_trace_max_step_count,
                positives,
                clip_ranges=self.clip_bounds
            )

        # Plot calculated lines

        with Timer("Plot Lines"):  # TODO - remove timers when ready
            self.__add_field_lines(field_lines, positives)

    def __add_field_lines(self,
                          lines: np.ndarray,
                          positives: np.ndarray) -> None:

        assert lines.shape[0] == positives.shape[0]

        for i in range(lines.shape[0]):
            points = lines[i]
            positive = positives[i]
            self.__add_field_line(points, positive)

    def __add_field_line(self,
                         points: np.ndarray,
                         positive: bool) -> None:

        self.switch_to()

        # If not enough points then don't draw anything

        if points.shape[0] <= 1:
            return

        # Draw parts of line

        prev = points[0]
        last_arrowhead_pos = prev/settings.VIEWPORT_SCALE_FAC  # N.B. not drawing arrohead at start

        for curr in points[1:]:

            # If point is repeated (probably meaning the line was clipped) then stop drawing
            if np.all(np.isclose(prev, curr)):
                break

            x1 = prev[0]/settings.VIEWPORT_SCALE_FAC
            y1 = prev[1]/settings.VIEWPORT_SCALE_FAC
            x2 = curr[0]/settings.VIEWPORT_SCALE_FAC
            y2 = curr[1]/settings.VIEWPORT_SCALE_FAC

            line = pyglet.shapes.Line(
                x1, y1,
                x2, y2,
                batch=self.field_lines_batch
            )
            self.__field_lines_shapes.add(line)

            screen_pos = np.array([x2,y2])

            if settings.show_field_line_arrows:

                if vectors.magnitudes(last_arrowhead_pos-screen_pos)[0] >= settings.field_line_render_arrowhead_spacing:

                    line_dir = (curr-prev) / vectors.magnitudes(curr-prev)[0] * (1 if positive else -1)
                    line_norm = np.array([line_dir[1], -line_dir[0]])
                    arrowhead_tip = screen_pos + (line_dir*ARROWHEAD_LENGTH)
                    arrowhead_side1 = screen_pos + (line_norm*ARROWHEAD_LENGTH/2)
                    arrowhead_side2 = screen_pos - (line_norm*ARROWHEAD_LENGTH/2)

                    arrowhead = pyglet.shapes.Triangle(
                        arrowhead_tip[0], arrowhead_tip[1],
                        arrowhead_side1[0], arrowhead_side1[1],
                        arrowhead_side2[0], arrowhead_side2[1],
                        batch=self.field_lines_batch
                    )
                    self.__field_lines_shapes.add(arrowhead)

                    last_arrowhead_pos = screen_pos

            prev = curr

    def clear_screen(self) -> None:

        self.__field_lines_shapes.clear()

    def round_float_pos(self, pos: np.ndarray) -> np.ndarray:

        return np.around(pos, decimals=0).astype(int)

    def on_draw(self) -> None:
        """Clear screen and draw batches of shapes"""

        self.clear()

        self.switch_to()

        self.field_lines_batch.draw()

        self.field_elements_batch.draw()

        if self.__click_mode_sprite is not None:
            self.__click_mode_sprite.draw()

    def update(self, delta_time: float) -> None:

        self.__lifetime += delta_time

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_press_callback(x, y, button, modifiers)

    def on_key_release(self, c, all_mods):

        # Only want to read the modifiers that might be part of a shortcut

        mods = 0

        if all_mods & pyglet.window.key.MOD_SHIFT:
            mods |= MOD_SHIFT

        if all_mods & pyglet.window.key.MOD_CTRL:
            mods |= MOD_CTRL

        if all_mods & pyglet.window.key.MOD_ALT:
            mods |= MOD_ALT

        # Run callback

        self.key_press_callback((c, mods))


class Controller:
    """A wrapper for a visualisation window for controlling the window"""

    event_loop: Optional[pyglet.app.EventLoop] = None

    def __init__(self,
                 window: Window,
                 field: Optional[Field] = None):

        self.__window = window

        if field is not None:
            self.__field = field
        else:
            self.__field = Field()

    def get_field(self) -> Field:
        return self.__field

    def set_field(self, field: Field) -> None:
        self.__field = field

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
        """Tries to remove an element at the screen position specified. Returns whether one was found"""

        for ele in self.__field.iter_elements():

            if _create_element_renderer(ele).point_in_draw_bounds(posx, posy):
                self.__field.remove_element(ele)
                self.redraw_only_elements()
                return True

        return False

    def set_click_mode_none(self):
        self.__window.set_click_mode_none()

    def set_click_mode_add(self):
        self.__window.set_click_mode_add()

    def set_click_mode_delete(self):
        self.__window.set_click_mode_delete()

    def activate_window(self):
        self.__window.activate()

    @staticmethod
    def run_app() -> None:
        """Starts the Pyglet event loop running.
Only should be called when all windows have been created and can't be called again from any instance of the class or otherwise.
Will block until all windows are closed.
"""

        if Controller.event_loop is not None:

            raise AppAlreadyRunningException()

        else:

            Controller.event_loop = pyglet.app.EventLoop()

            Controller.event_loop.run()

    @staticmethod
    def quit_app() -> None:

        if Controller.event_loop is None:

            raise AppNotRunningException()

        else:

            Controller.event_loop.has_exit = True


def create_window(
        on_exit: Callable[[], None],
        on_mouse_press: Callable[[int, int, int, int], None],
        on_key_press: Callable[[KeyPressCommand], None]
    ) -> Controller:
    """Create and open the visualisation window.
Note that this doesn't start the window running

Parameters:

    on_exit - a callable run when the window is closed

    on_mouse_press - a callable run when the window is clicked on with the mouse. \
The inputs to the callable are the same as those for pyglet.Window.on_mouse_press (x, y, button, modifiers)

    on_key_press - a callable run when a key is pressed with the window in focus

Returns:

    controller - a controller object for the window
"""

    # Create main window
    window = Window(
        WINDOW_DEFAULT_WIDTH,
        WINDOW_DEFAULT_HEIGHT,
        on_exit=on_exit,
        on_mouse_press=on_mouse_press,
        on_key_press=on_key_press
    )

    controller = Controller(window)  # Create window's controller

    # Schedule window's update function

    pyglet.clock.schedule_interval(window.update, 1/30)

    # Return the window controller

    return controller


if __name__ == "__main__":

    controller = create_window(
        on_exit=lambda: None,
        on_mouse_press=lambda x, y, btn, mods: print(f"Click at {x}, {y} with button {btn}"),
        on_key_press=lambda cmd: print(f"Command {cmd} pressed")
    )
    controller.run_app()
