from decimal import Decimal
from typing import Literal

from collections.abc import Callable


def get_trendline(function: Callable[[float], Decimal], data: dict[int, dict[Literal["mean", "std_dev"], Decimal]]) -> dict[int, Decimal]:
    """
    Generates best matched trendline for given data that is calculated from given function formula.

    Params:
    - `function` (`Callable[[int], Decimal]`): Function formula.
    - `data` (`dict[int, dict[Literal["mean", "std_dev"], Decimal]]`): Data to be matched.

    Return
    - `dict[int, Decimal]`: Trendline definition where keys are instance sizes and values are time measurement results
    for them.
    """
    print("Generating trend line...")
    ERROR_TOLERANCE: float = 1e-6
    def __mean_relative_error(expected: dict[int, Decimal], actual: dict[int, Decimal]) -> Decimal:
        mean_error: Decimal = Decimal(0)
        counter: int = 0
        for expected_value, actual_value, size in zip(expected.values(), actual.values(), actual.keys()):
            mean_error += (
                (Decimal(actual_value) - Decimal(expected_value)) / Decimal(expected_value) 
                * Decimal(size / max(actual.keys()))
            )
            counter += 1
        return mean_error / counter
    
    constant: Decimal = Decimal(1)
    while True:
        trendline: dict[int, Decimal] = {size: Decimal( constant * function(size)) for size in data.keys()}
        mean_relative_error: Decimal = __mean_relative_error(
            {size: result["mean"] for size, result in data.items()}, trendline
        )
        if abs(mean_relative_error) <= ERROR_TOLERANCE:
            break
        constant += constant * Decimal(0.01) * (Decimal(1) if mean_relative_error < 0 else Decimal(-1))

    print(f"Mean relative error for trend line: {mean_relative_error}")
    return {
        size: constant * function(size) for size in range(min(data.keys()), max(data.keys()) + 1, 10)
    }