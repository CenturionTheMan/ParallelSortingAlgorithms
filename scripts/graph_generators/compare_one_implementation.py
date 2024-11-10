import sys
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure

import _common.constants as const
import _common.loader as loader
import _common.trendline as trendline
import _common.saver as saver


MENU_OPTIONS: dict[str, dict[str, str | const.Implementation]] = {
    "1": "bitonic sort (CPU)",
    "2": "bitonic sort (GPU)",
    "3": "odd-even sort (CPU)",
    "4": "odd-event sort (GPU)"
}


print("Choose implementation by entering its corresponding number:")
for option_id, label in MENU_OPTIONS.items():
    print(f"{option_id}: {label}")
user_choice: str = input(">")

if user_choice not in MENU_OPTIONS:
    print("Invalid option! Generator would be terminated!")
    input("Press ENTER to continue...")
    exit(1)

result_1: dict[const.Implementation, dict[int, float]] = {
    implementation: results for implementation, results in loader.load_results_csv(sys.argv[1]).items()
    if implementation == MENU_OPTIONS[user_choice]
}
result_2: dict[const.Implementation, dict[int, float]] = {
    implementation: results for implementation, results in loader.load_results_csv(sys.argv[2]).items()
    if implementation == MENU_OPTIONS[user_choice]
}

graph: tuple[Figure, plt.Axes] = plt.subplots(figsize=(const.A4_WIDTH_INCHES, const.A4_WIDTH_INCHES))
graph[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
graph[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="x", scilimits=(0,0))

for result, color in zip([result_1, result_2], ["r", "b"]):
    for implementation, results_1 in result.items():
        trend_line: dict[float, float] = trendline.get_trendline(const.TREND_LINES[implementation], results_1)
        graph[1].plot(
            results_1.keys(),
            results_1.values(),
            f"{const.MARKERS[implementation]}{color}",
            label=f"Pomiar {implementation}"
        )
        graph[1].plot(
            trend_line.keys(), 
            trend_line.values(), f"--{color}", 
            label=f"Krzywa wzorcowa {const.TREND_LINES_NAMES[implementation]}: {implementation}"
        )

graph[1].legend()
plt.title("\n".join(
    wrap(f"Porównanie wyników dwóch pomiarów implementacji {MENU_OPTIONS[user_choice]}")
))
plt.xticks(rotation=90)
plt.xlabel(r"Rozmiar instancji [$n$]")
plt.ylabel("Czas sortowania [s]")
plt.xticks()
plt.grid(visible=True)
print(f"Graph comparing time compelxity for two {MENU_OPTIONS[user_choice]} implementation has been generated!")
plt.show()

saver.save_plot_as_pdf(graph[0], f"Compare {MENU_OPTIONS[user_choice]}")