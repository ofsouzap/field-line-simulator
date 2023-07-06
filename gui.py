import pyglet
from typing import List, Tuple
from os.path import join as joinpath
from field import Field
from point_source import PointSource
import numpy as np
from _debug_util import Timer


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


DIAGRAM_WINDOW_CAPTION = "Field Line Simulator - Diagram"
DIAGRAM_WINDOW_WIDTH = 720
DIAGRAM_WINDOW_HEIGHT = 480
DIAGRAM_WINDOW_BACKGROUND_COLOR = BLACK


CONTROL_BUTTON_ICON_RES_DIR = "icons"
CONTROLS_WINDOW_CAPTION = "Field Line Simulator - Controls"

CONTROL_BUTTON_DETAILS: List[Tuple[str, str]] = [
    ("Save (Ctrl+S)", "save.png"),
    ("Open (Ctrl+O)", "load.png"),
    ("Add (A)", "placeholder.png"),
    ("Delete (D)", "placeholder.png"),
    ("Settings (S)", "placeholder.png"),
    ("Help", "placeholder.png"),
    ("Recalculate (R)", "placeholder.png"),
]
"""Details about each controls window button.

Tuple values:

    label - the label of the button

    icon_res - the image file for the image's icon
"""
CONTROL_BUTTON_COUNT = len(CONTROL_BUTTON_DETAILS)

CONTROL_BUTTON_WIDTH = 90
CONTROL_BUTTON_HEIGHT = 90
CONTROL_BUTTON_IMAGE_LENGTH = 45
"""The side length of the control buttons' images"""
CONTROL_BUTTON_PADDING = 10
"""How many pixels to have on the left, right, top and bottom of the buttons to space them out"""

CONTROLS_WINDOW_BACKGROUND_COLOR = BLACK
CONTROL_BUTTON_BACKGROUND_COLOR = (128, 128, 128, 255)
CONTROL_BUTTON_LABEL_COLOR = WHITE

CONTROL_BUTTON_LABEL_FONT_SIZE = 8

CBTN_SAVE = 0
CBTN_LOAD = 1
CBTN_ADD = 2
CBTN_DELETE = 3
CBTN_SETTINGS = 4
CBTN_HELP = 5
CBTN_RECALCULATE = 6


class DiagramWindow(pyglet.window.Window):

    def __init__(self, width: int, height: int):

        super().__init__(width, height, DIAGRAM_WINDOW_CAPTION)

        self.batch = pyglet.graphics.Batch()

        self.__lifetime = 0
        self.__field_line_lines = []

    def draw_field(self,
                   field: Field) -> None:
        """Clears the current diagram and draws the field provided"""

        self.switch_to()

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

        self.switch_to()

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

        self.switch_to()

        self.__field_line_lines.clear()

    def round_float_pos(self, pos: np.ndarray) -> np.ndarray:

        return np.around(pos, decimals=0).astype(int)

    def on_draw(self) -> None:
        """Clear screen and draw batch of shapes"""

        pyglet.graphics.glClearColor(*[v/255 for v in DIAGRAM_WINDOW_BACKGROUND_COLOR])
        self.clear()
        self.batch.draw()

    def update(self, delta_time: float) -> None:

        self.__lifetime += delta_time


class ControlsWindow(pyglet.window.Window):

    def __init__(self):

        super().__init__(
            1,
            1,
            CONTROLS_WINDOW_CAPTION
        )

        # Create record of largest indexes used

        self.__max_btn_index_x = 0
        self.__max_btn_index_y = 0

        # Create batch

        self.batch = pyglet.graphics.Batch()

        # Create saved drawables

        self.__drawables = set()

        # Create buttons

        self.__create_button_by_id(CBTN_SAVE, 0, 0)
        self.__create_button_by_id(CBTN_LOAD, 0, 1)
        self.__create_button_by_id(CBTN_ADD, 1, 0)
        self.__create_button_by_id(CBTN_DELETE, 1, 1)
        self.__create_button_by_id(CBTN_SETTINGS, 2, 0)
        self.__create_button_by_id(CBTN_HELP, 2, 1)
        self.__create_button_by_id(CBTN_RECALCULATE, 2, 2)

        # Update window width and height

        self.width = (CONTROL_BUTTON_WIDTH + (2 * CONTROL_BUTTON_PADDING)) * (self.__max_btn_index_x + 1)
        self.height = (CONTROL_BUTTON_HEIGHT + (2 * CONTROL_BUTTON_PADDING)) * (self.__max_btn_index_y + 1)

    @staticmethod
    def load_button_icon(icon_res: str):
        return pyglet.resource.image(
            joinpath(
                CONTROL_BUTTON_ICON_RES_DIR,
                icon_res),
            atlas=False
        )

    def __create_button_by_id(self,
                              btn_id: int,
                              index_x: int,
                              index_y: int) -> None:

            btn_label, btn_image_path = CONTROL_BUTTON_DETAILS[btn_id]

            self.__create_controls_button(
                btn_label=btn_label,
                btn_image_path=btn_image_path,
                image_length=CONTROL_BUTTON_IMAGE_LENGTH,
                width=CONTROL_BUTTON_WIDTH,
                height=CONTROL_BUTTON_HEIGHT,
                index_x=index_x,
                index_y=index_y,
                padding=CONTROL_BUTTON_PADDING,
                background_color=CONTROL_BUTTON_BACKGROUND_COLOR
            )

    def __create_controls_button(self,
                                 btn_label: str,
                                 btn_image_path: str,
                                 image_length: int,
                                 width: int,
                                 height: int,
                                 index_x: int,
                                 index_y: int,
                                 padding: int = 0,
                                 label_color: Tuple[int, int, int, int] = WHITE,
                                 background_color: Tuple[int, int, int, int] = CONTROLS_WINDOW_BACKGROUND_COLOR) -> None:
        """Creates a button in this controls window.

Parameters:

btn_label - the label to put for the button

btn_image_path - the path for the button's image relative to CONTROLS_ICONS_RESOURCES_DIR

image_length - the image will be drawn as a square. This is the side length that this square should have

width - the width of the area that a button takes up. This should be greater than the button image's width

height - the height of the area that a button takes up. This should be greater than the button image's height

index_x - the x-index of the button. This is used to calculate the position of the button

index_y - the y-index of the button. This is used to calculate the position of the button

padding - the number of pixels of padding to give each button

label_color - what color to draw the label in

background_color - background color for the buttons
"""

        self.switch_to()

        if index_x > self.__max_btn_index_x:
            self.__max_btn_index_x = index_x

        if index_y > self.__max_btn_index_y:
            self.__max_btn_index_y = index_y

        # Calculate button positions

        left = padding + ((width + (2*padding)) * index_x)
        right = left + width
        mid_x = (left + right) // 2

        bottom = padding + ((height + (2*padding)) * index_y)
        top = bottom + height

        image_top = top
        image_bottom = image_top - image_length
        image_mid_y = (image_top + image_bottom) // 2

        label_top = image_bottom
        label_bottom = bottom
        label_mid_y = (label_top + label_bottom) // 2

        # Background

        background_rect = pyglet.shapes.Rectangle(
            x=left,
            y=bottom,
            width=width,
            height=height,
            color=background_color,
            batch=self.batch
        )

        self.__drawables.add(background_rect)

        # Label

        label = pyglet.text.Label(
            text=btn_label,
            font_size=CONTROL_BUTTON_LABEL_FONT_SIZE,
            x=mid_x,
            y=label_mid_y,
            z=1,
            anchor_x="center",
            anchor_y="center",
            color=label_color,
            align="center",
            multiline=True,
            width=width,
            batch=self.batch
        )

        self.__drawables.add(label)

        # Icon

        image = ControlsWindow.load_button_icon(btn_image_path)

        image.height = image_length
        image.width = image_length

        img_width_half = image.width // 2
        img_height_half = image.height // 2

        image_sprite = pyglet.sprite.Sprite(
            image,
            x=mid_x-img_width_half,
            y=image_mid_y-img_height_half,
            z=1,
            batch=self.batch
        )

        self.__drawables.add(image_sprite)

    def update(self, delta_time: float) -> None:
        pass

    def on_draw(self) -> None:

        pyglet.graphics.glClearColor(*[v/255 for v in CONTROLS_WINDOW_BACKGROUND_COLOR])
        self.clear()
        self.batch.draw()


def open_gui():
    """Create and open the diagram GUI and control GUI. Blocks until the windows are closed"""

    diagram_window = DiagramWindow(DIAGRAM_WINDOW_WIDTH, DIAGRAM_WINDOW_HEIGHT)  # Create diagram window
    controls_window = ControlsWindow()  # Create controls window

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

    diagram_window.switch_to()

    for i, s in enumerate(sources_dat):
        source_shapes.append(pyglet.shapes.Circle(s[0], s[1], 3, batch=diagram_window.batch))

        field.add_element(PointSource(np.array([s[0], s[1]]), s[2]))

    diagram_window.draw_field(field)

    # Register update functions

    # pyglet.clock.schedule_interval(diagram_window.update, 1/30)
    # pyglet.clock.schedule_interval(controls_window.update, 1/30)

    # Run app

    pyglet.app.run()


if __name__ == "__main__":
    open_gui()
