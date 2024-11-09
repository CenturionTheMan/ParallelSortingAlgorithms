import math
from collections.abc import Callable
from typing import Literal, TypeAlias


Implementation: TypeAlias = Literal["bitonic sort (CPU)", "bitonic sort (GPU)", "odd-even sort (CPU)", "odd-event sort (GPU)"]


MARKERS: dict[Implementation, str] = {
    "bitonic sort (CPU)": "p", 
    "bitonic sort (GPU)": "o", 
    "odd-even sort (CPU)": "s", 
    "odd-event sort (GPU)": "^"
}


COLORS: dict[Implementation, str] = {
    "bitonic sort (CPU)": "b", 
    "bitonic sort (GPU)": "g", 
    "odd-even sort (CPU)": "r", 
    "odd-event sort (GPU)": "c"
}


TREND_LINES: dict[Implementation, Callable[[float], float]] = {
    "bitonic sort (CPU)": lambda size: math.log2(size) ** 2, 
    "bitonic sort (GPU)": lambda size: math.log2(size) ** 2, 
    "odd-even sort (CPU)": lambda size: float(size), 
    "odd-event sort (GPU)": lambda size: float(size)
}


TREND_LINES_NAMES: dict[Implementation, str] = {
    "bitonic sort (CPU)": r"$\log^2(n)$", 
    "bitonic sort (GPU)": r"$\log^2(n)$", 
    "odd-even sort (CPU)": r"$n$", 
    "odd-event sort (GPU)": r"$n$"
}


A4_WIDTH_INCHES: float = 8.27


PLOTS_DIR_PATH: str = "gfx/plots"