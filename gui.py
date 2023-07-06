import pyglet
from field import Field, List
from point_source import PointSource
import numpy as np
from _debug_util import Timer
from typing import Callable, Set, Any, List, Tuple
from pathlib import Path
from os.path import join as joinpath


BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


WINDOW_TITLE = "Field Line Simulator"
WINDOW_DEFAULT_WIDTH = 720
WINDOW_DEFAULT_HEIGHT = 480


CONTROLS_ICONS_RESOURCES_DIR = "icons"
"""The resources subdirectory containing images for the controls buttons' icons"""


CONTROLS_IMAGE_LENGTH: int = 64
CONTROLS_BAR_BUTTON_WIDTH: int = 100
CONTROLS_BAR_HEIGHT: int = 100


CONTROLS: List[Tuple[str, str]] = [
    ("Save (Ctrl+S)", "save.png"),
    ("Open (Ctrl+O)", "load.png"),
    ("...", "placeholder.png"),
    ("...", "placeholder.png"),
    ("...", "placeholder.png"),
    ("...", "placeholder.png"),
    ("...", "placeholder.png"),
]
"""Details about each of the control options in the controls bar for the gui.

Each tuple is:

    label - the label for the control

    image - the resource path for the control's icon relative to CONTROLS_ICONS_RESOURCES_DIR
"""


def controls_icon(image_path: str) -> pyglet.image.TextureRegion:
    return pyglet.resource.image(joinpath(CONTROLS_ICONS_RESOURCES_DIR, image_path), atlas=False)


class MainWindow(pyglet.window.Window):

    def __init__(self, width: int, height: int):

        super().__init__(width, height, WINDOW_TITLE)

        # Create graphics batches

        self.diagram_batch = pyglet.graphics.Batch()
        self.controls_batch = pyglet.graphics.Batch()

        # Create members

        self.__lifetime = 0
        self.__field_line_lines = []

        self.__controls_drawables: Set[Any] = set()

        # Create controls panel

        for i, (btn_label, btn_image) in enumerate(CONTROLS):

            self.__create_controls_button(
                btn_label=btn_label,
                btn_image_path=btn_image,
                on_press=lambda: print(f"Button \"{btn_label}\" pressed"),  # TODO - do actual functionalities
                image_length=CONTROLS_IMAGE_LENGTH,
                width=CONTROLS_BAR_BUTTON_WIDTH,
                height=CONTROLS_BAR_HEIGHT,
                index = i
            )

    def __create_controls_button(self,
                                 btn_label: str,
                                 btn_image_path: str,
                                 on_press: Callable[[], None],
                                 image_length: int,
                                 width: int,
                                 height: int,
                                 index: int) -> None:
        """Creates a button in the controls bar that is rendered when the window is drawn

Parameters:

    btn_label - the label to put for the button

    btn_image_path - the path for the button's image relative to CONTROLS_ICONS_RESOURCES_DIR

    on_press - the callback to call when the button is pressed

    image_length - the image will be drawn as a square. This is the side length that this square should have

    width - the width of the area that a button takes up. This should be greater than the button image's width

    height - the height of the area that a button takes up. This should be greater than the button image's height

    index - the index of the button. This is used to calculate the position of the button's image and label
"""

        # Calculate button positions

        left = width * index
        right = left + width
        mid_x = left + (width // 2)

        top = self.height
        bottom = top - height

        image_top = top
        image_bottom = image_top - image_length
        image_mid_y = (image_top + image_bottom) // 2

        label_top = image_bottom
        label_bottom = bottom
        label_mid_y = (label_top + label_bottom) // 2

        # Create label and image

        label = pyglet.text.Label(
            text=btn_label,
            x=mid_x,
            y=label_mid_y,
            anchor_x="center",
            anchor_y="center",
            color=WHITE,
            align="center",
            multiline=True,
            width=CONTROLS_BAR_BUTTON_WIDTH,
            batch=self.controls_batch
        )

        self.__controls_drawables.add(label)

        image = controls_icon(btn_image_path)
        img_width_half = image.width // 2
        img_height_half = image.height // 2
        image_sprite = pyglet.sprite.Sprite(image, x=mid_x-img_width_half, y=image_mid_y-img_height_half, batch=self.controls_batch)

        self.__controls_drawables.add(image_sprite)

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
                batch=self.diagram_batch
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

        self.diagram_batch.draw()
        self.controls_batch.draw()

    def update(self, delta_time: float) -> None:

        self.__lifetime += delta_time


def open_gui():
    """Create and open the main GUI. Blocks until the window is closed"""

    window = MainWindow(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)  # Create main window

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

        source_shapes.append(pyglet.shapes.Circle(s[0], s[1], 3, batch=window.diagram_batch))

        field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

    window.draw_field(field)

    pyglet.clock.schedule_interval(window.update, 1/30)  # Register window's update function

    pyglet.app.run()  # Run app


if __name__ == "__main__":
    open_gui()
