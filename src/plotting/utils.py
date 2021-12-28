import colorsys
import re
from typing import Dict, Tuple
from beartype import beartype


@beartype
def raise_for_incorrect_color(palette: Dict[str, str]) -> None:
    """Raises exception if color check has failed.
    :param palette: Dict[str, str], palette with HEX colors.
    :return:
    """
    for object_type, color in palette.items():
        found_color = re.search(r"^#[0-9a-fA-F]{6}$", color)

        if found_color is None:
            raise ValueError(f"Color for {object_type} wasn't recognized: {color}.")
    return


@beartype
def change_lightness(color: Tuple[int, int, int],
                     shift: float) -> Tuple[int, int, int]:
    """Changes color lightness depending on shift parameter.
    :param color: Tuple[int, int, int], RGB tuple of uint8.
    :param shift: float, value in range from zero to one.
                  If it's less than 0.5 output color will
                  have decreased lightness, and increased
                  lightness otherwise.
    :return: Tuple[int, int, int], shifted color, RGB tuple of uint8.
    """
    if not isinstance(color, tuple) or len(color) != 3:
        raise TypeError(
            "Wrong type of input color! Color must be (int, int, int) with values between 0 and 255.",
        )

    if not all(isinstance(c, int) for c in color):
        raise TypeError("Color has wrong value types!")

    if max(color) > 255 or min(color) < 0:
        raise ValueError("Color has wrong values!")

    if shift > 1.0 or shift < 0.0:
        raise ValueError(f"Wrong value for shift parameter. Value must be between 0 and 1, now {shift}.")

    float_color = (channel / 255 for channel in color)
    hls_color = colorsys.rgb_to_hls(*float_color)
    hue, lightness, saturation = hls_color

    if shift > 0.5:
        space = 1.0 - lightness
    else:
        space = lightness

    lightness += (shift - 0.5) * 2 * space

    rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(int(channel * 255) for channel in rgb_color)
