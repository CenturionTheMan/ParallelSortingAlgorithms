import csv

import _common.constants as const


def load_results_csv(
    csv_file: str
) -> dict[const.Implementation, dict[int, float]]:
    """
    Loads all measurement results from file with given path.

    Params
    - `csv_file` (`str`): Path to a CSV file with measurement results.

    Return
    - `dict[dict[int, float]]`: Dictionary of dictionaries where each nested dictionary corresponds to one of measured 
    implementations. Nested dictionaries include key-value pairs wieher key is an instance size and value is a floating
    point measurement result for it. Example:

        ```
        {
            "odd even gpu": {
                1: 1.234,
                2: 34.67
            }
        }
        ```
    """
    result: dict[const.Implementation, dict[int, float]] = {
        "bitonic sort (CPU)": {},
        "bitonic sort (GPU)": {},
        "odd-even sort (CPU)": {},
        "odd-event sort (GPU)": {}
    }
    repetitions: dict[const.Implementation, dict[int, int]] = {
        "bitonic sort (CPU)": {},
        "bitonic sort (GPU)": {},
        "odd-even sort (CPU)": {},
        "odd-event sort (GPU)": {}
    }
    with open(csv_file, "r") as file:
        reader: csv.DictReader = csv.DictReader(file, delimiter=";")
        for record in reader:
            for implementation in result.keys():
                if record["instance size"] not in result[implementation]:
                    result[implementation][int(record["instance size"])] = 0.0
                    repetitions[implementation][record["instance size"]] = 0
                result[implementation][int(record["instance size"])] += float(record[implementation])
                repetitions[implementation][record["instance size"]] += 1

    not_measured_implementations: set = set()
    for implementation_, results in result.items():
        for instance_size, time_measurement in results.items():
            time_measurement /= repetitions[implementation_][str(instance_size)]
            if time_measurement == 0.0:
                not_measured_implementations.add(implementation_)

    return {
        implementation: results 
        for implementation, results in result.items() if implementation not in not_measured_implementations
    }