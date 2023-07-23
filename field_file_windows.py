from typing import Optional
from tkinter import filedialog
from field import Field
from field import FieldSerialize


FIELD_FILE_FILETYPE: str = "*.field"


def save_field(field: Field) -> None:

    file = filedialog.asksaveasfile(
        mode="w",
        defaultextension=FIELD_FILE_FILETYPE,
        filetypes=[("Field file", FIELD_FILE_FILETYPE), ("All files", "*.*")]
    )

    if file is not None:

        FieldSerialize.serialize(field, file)  # type: ignore

        file.close()


def load_field() -> Optional[Field]:

    file = filedialog.askopenfile(
        mode="r",
        filetypes=[("Field file", FIELD_FILE_FILETYPE), ("All files", "*.*")]
    )

    if file is not None:

        field = FieldSerialize.deserialize(file)  # type: ignore

        file.close()

        return field

    else:

        return None
