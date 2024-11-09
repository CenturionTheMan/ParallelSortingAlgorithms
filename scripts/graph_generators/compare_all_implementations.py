import sys
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure

import _common.constants as const
import _common.loader as loader
import _common.trendline as trendline
import _common.saver as saver


results: dict[const.Implementation, dict[int, float]] = loader.load_results_csv(sys.argv[1])

graph: tuple[Figure, plt.Axes] = plt.subplots(figsize=(const.A4_WIDTH_INCHES, const.A4_WIDTH_INCHES))
graph[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
graph[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
graph[1].ticklabel_format(style="sci", axis="x", scilimits=(0,0))

for implementation, results in results.items():
    trend_line: dict[int, float] = trendline.get_trendline(const.TREND_LINES[implementation], results)
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

graph[1].legend()
plt.title("\n".join(
    wrap("Porównanie złożoności czasowych wszystkich implementacji algorytmów bitonic sort oraz odd-even sort")
))
plt.xticks(rotation=90)
plt.xlabel(r"Rozmiar instancji [$n$]")
plt.ylabel("Czas sortowania [s]")
plt.xticks()
plt.grid(visible=True)
print("Graph for comapring all implementations generated!")
plt.show()

saver.save_plot_as_pdf(graph[0], "Implementations comparison")