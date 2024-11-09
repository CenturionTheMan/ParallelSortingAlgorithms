import sys
from collections.abc import Callable
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure

import _common.constants as const
import _common.loader as loader
import _common.trendline as trendline
import _common.saver as saver


MENU_OPTIONS: dict[str, dict[str, str | list[const.Implementation]]] = {
    "1": ["bitonic sort (CPU)", "bitonic sort (GPU)"],
    "2": ["odd-even sort (CPU)", "odd-event sort (GPU)"],
}


print("Choose algorithm by entering its corresponding number:")
for option_id, implementations in MENU_OPTIONS.items():
    print(f"{option_id}: {" ".join(implementations[0].split(" ")[:-1])}")
user_choice: str = input(">")

if user_choice not in MENU_OPTIONS:
    print("Invalid option! Generator would be terminated!")
    input("Press ENTER to continue...")
    exit(1)

results: dict[const.Implementation, dict[int, float]] = {
    implementation: results for implementation, results in loader.load_results_csv(sys.argv[1]).items()
    if implementation in MENU_OPTIONS[user_choice]
}

graph: tuple[Figure, plt.Axes] = plt.subplots(figsize=(const.A4_WIDTH_INCHES, const.A4_WIDTH_INCHES))
graph[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
graph[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="x", scilimits=(0,0))

for implementation, results in results.items():
    trend_line: dict[float, float] = trendline.get_trendline(const.TREND_LINES[implementation], results)
    graph[1].plot(
        results.keys(),
        results.values(),
        f"{const.MARKERS[implementation]}{const.COLORS[implementation]}",
        label=f"Pomiar {implementation}"
    )
    graph[1].plot(
        trend_line.keys(), 
        trend_line.values(), f"--{const.COLORS[implementation]}", 
        label=f"Krzywa wzorcowa {const.TREND_LINES_NAMES[implementation]}: {implementation}"
    )

get_algorithm_name: Callable[[], str] = lambda: " ".join(MENU_OPTIONS[user_choice][0].split(" ")[:-1])

graph[1].legend()
plt.title("\n".join(
    wrap(f"Porównanie złożoności czasowych implementacji CPU i GPU algorytmu {get_algorithm_name()}")
))
plt.xticks(rotation=90)
plt.xlabel(r"Rozmiar instancji [$n$]")
plt.ylabel("Czas sortowania [s]")
plt.xticks()
plt.grid(visible=True)
print(f"Time compelxity comparison for implementations of {get_algorithm_name()} algorithm has been generated!")
plt.show()

saver.save_plot_as_pdf(graph[0], f"GPU CPU comparison {get_algorithm_name()}")