import csv
import math
from decimal import Decimal
from typing import Literal

import _common.constants as const


def __calculate_mean_and_std_dev(results: list[Decimal]) -> dict[Literal["mean", "std_dev"], Decimal]:
    """
    Calculates mean and standard derivation for given results.
    """
    average_result: dict[Literal["mean", "std_dev"], Decimal] = {
        "mean": Decimal(0),
        "std_dev": Decimal(0),
    }
    for result in results:
        average_result["mean"] += result
    average_result["mean"] /= len(results)
    
    for result in results:
        average_result["std_dev"] += (result - average_result["mean"]) ** 2
    average_result["std_dev"] /= len(results)
    average_result["std_dev"] = Decimal(math.sqrt(average_result["std_dev"]))
    
    return average_result


def load_results_csv(
    csv_file: str
) -> dict[const.Implementation, dict[int, dict[Literal["mean", "std_dev"], Decimal]]]:
    """
    Loads all measurement results from file with given path.

    Params
    - `csv_file` (`str`): Path to a CSV file with measurement results.

    Return
    - `dict[dict[int, Decimal]]`: Dictionary of dictionaries where each nested dictionary corresponds to one of measured 
    implementations. Nested dictionaries include key-value pairs wieher key is an instance size and value is a floating
    point measurement result and standard derivation for it. Example:

        ```
        {
            "odd even gpu": {
                1: 1.234,
                2: 34.67
            }
        }
        ```
    """
    print("Loading results...")
    result: dict[const.Implementation, dict[int, dict[Literal["mean", "std_dev"], Decimal]]] = {
        "bitonic sort (CPU)": {},
        "bitonic sort (GPU)": {},
        "odd-even sort (CPU)": {},
        "odd-event sort (GPU)": {}
    }
    repetitions_results: dict[const.Implementation, dict[int, list[Decimal]]] = {
        "bitonic sort (CPU)": {},
        "bitonic sort (GPU)": {},
        "odd-even sort (CPU)": {},
        "odd-event sort (GPU)": {}
    }
    with open(csv_file, "r") as file:
        reader: csv.DictReader = csv.DictReader(file, delimiter=";")
        for record in reader:
            for implementation in repetitions_results.keys():
                if int(record["instance size"]) not in repetitions_results[implementation]:
                    repetitions_results[implementation][int(record["instance size"])] = []
                result_from_file: Decimal = (
                    Decimal(record[f"mean {implementation}"])
                    if record[f"mean {implementation}"] != ""
                    else Decimal(0.0)
                )
                repetitions_results[implementation][int(record["instance size"])].append(result_from_file)

    not_measured_implementations: set = set()
    for implementation_, results in repetitions_results.items():
        for instance_size, time_measurements in results.items():
            result[implementation_][instance_size] = __calculate_mean_and_std_dev(time_measurements)
            if result[implementation_][instance_size]["mean"] == 0.0:
                not_measured_implementations.add(implementation_)

    return {
        implementation: results 
        for implementation, results in result.items() if implementation not in not_measured_implementations
    }