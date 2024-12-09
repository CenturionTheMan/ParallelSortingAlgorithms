import math
from collections.abc import Callable
from decimal import Decimal
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


TREND_LINES: dict[Implementation, Callable[[int], Decimal]] = {
    "bitonic sort (CPU)": lambda size: Decimal(size), 
    "bitonic sort (GPU)": lambda size: Decimal(math.log2(Decimal(size))) ** 2,
    "odd-even sort (CPU)": lambda size: Decimal(size) ** 2,
    "odd-event sort (GPU)": lambda size: Decimal(size) ** 2
}


TREND_LINES_NAMES: dict[Implementation, str] = {
    "bitonic sort (CPU)": r"$n$", 
    "bitonic sort (GPU)": r"$\log^2{n}$", 
    "odd-even sort (CPU)": r"$n^2$", 
    "odd-event sort (GPU)": r"$n^2$"
}


A4_WIDTH_INCHES: float = 8.27


PLOTS_DIR_PATH: str = "gfx/plots"