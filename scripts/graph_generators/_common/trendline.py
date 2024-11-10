from decimal import Decimal

from collections.abc import Callable


def get_trendline(function: Callable[[float], Decimal], data: dict[int, Decimal]) -> dict[int, Decimal]:
    """
    Generates best matched trendline for given data that is calculated from given function formula.

    Params:
    - `function` (`Callable[[int], Decimal]`): Function formula.
    - `data` (`dict[int, Decimal]`): Data to be matched.

    Return
    - `dict[int, Decimal]`: Trendline definition where keys are instance sizes and values are time measurement results
    for them.
    """
    ERROR_TOLERANCE: float = 1e-6
    def __mean_relative_error(expected: dict[int, Decimal], actual: dict[float, Decimal]) -> Decimal:
        mean_error: Decimal = Decimal(0)
        counter: int = 0
        for expected_value, actual_value in zip(expected.values(), actual.values()):
            mean_error += (Decimal(actual_value) - Decimal(expected_value)) / Decimal(expected_value)
            counter += 1
        return mean_error / counter
    
    constant: Decimal = Decimal(1)
    while True:
        trendline: dict[int, Decimal] = {size: Decimal( constant * function(size)) for size in data.keys()}
        mean_relative_error: Decimal = __mean_relative_error(data, trendline)
        if abs(mean_relative_error) <= ERROR_TOLERANCE:
            break
        constant += constant * Decimal(0.01) * (Decimal(1) if mean_relative_error < 0 else Decimal(-1))

    print(mean_relative_error)
    return {
        size: constant * function(size) for size in range(min(data.keys()), max(data.keys()) + 1)
    }