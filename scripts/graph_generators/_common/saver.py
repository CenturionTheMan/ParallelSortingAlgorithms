import os

from matplotlib.figure import Figure

import _common.constants as const


def save_plot_as_pdf(figure: Figure, plot_name: str):
    def __build_plot_filename() -> str:
        plot_number: int = 0
        while os.path.exists(f"{const.PLOTS_DIR_PATH}/{plot_name} ({plot_number}).pdf"):
            plot_number += 1
        return f"{const.PLOTS_DIR_PATH}/{plot_name} ({plot_number}).pdf"
        

    if not os.path.exists(const.PLOTS_DIR_PATH):
        os.mkdir(const.PLOTS_DIR_PATH)
    
    figure.savefig(__build_plot_filename(), format="pdf")
    print(f"Graph saved to {__build_plot_filename()}!")