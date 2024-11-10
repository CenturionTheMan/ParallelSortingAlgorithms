from collections.abc import Callable


def get_trendline(function: Callable[[float], float], data: dict[int, float]) -> dict[int, float]:
    """
    Generates best matched trendline for given data that is calculated from given function formula.

    Params:
    - `function` (`Callable[[int], float]`): Function formula.
    - `data` (`dict[int, float]`): Data to be matched.

    Return
    - `dict[int, float]`: Trendline definition where keys are instance sizes and values are time measurement results
    for them.
    """
    ERROR_TOLERANCE: float = 1e-3
    def __mean_relative_error(expected: dict[int, float], actual: dict[float, float]) -> float:
        mean_error = 0
        counter = 0
        for expected_value, actual_value in zip(expected.values(), actual.values()):
            mean_error += (actual_value - expected_value) / expected_value
            counter += 1
        return mean_error / counter
    
    constant: float = 1
    while True:
        trendline: dict[int, float] = {size: constant * function(size) for size in data.keys()}
        mean_relative_error: float = __mean_relative_error(data, trendline)
        if abs(mean_relative_error) <= ERROR_TOLERANCE:
            break
        constant += constant * 0.01 * (1 if mean_relative_error < 0 else -1)
    
    def __generate_trendline_arguments(first: int, last: int) -> list[float]:
        arguments: list[float] = []
        argument: float = float(first)
        while argument <= last:
            arguments.append(argument)
            argument += 0.1
        return arguments

    return {
        size: constant * function(size) for size in __generate_trendline_arguments(min(data.keys()), max(data.keys()))
    }