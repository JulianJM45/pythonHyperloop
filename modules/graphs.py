import os

import matplotlib.pyplot as plt
import numpy as np

# set font type
plt.rcParams["font.sans-serif"] = "cmss10"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

OUTPUT_FOLDER = "../../MA_thesis_typst/figures/"


class Graph:
    def __init__(self, width_cm=15, height_cm=None, fontsize=9):
        self.width_cm = width_cm
        self.height_cm = height_cm if height_cm else width_cm * (9 / 16)
        self.fontsize = fontsize
        self.polar = False
        self.colormap = False
        self.ax = None

        width_in = self.width_cm / 2.54
        height_in = self.height_cm / 2.54
        plt.figure(figsize=(width_in, height_in), dpi=100)

    def go_polar(self):
        self.polar = True
        self.ax = plt.subplot(111, projection="polar")

    def add_plot(self, x, y, label="", color="red", linewidth=1):
        if self.polar:
            self.ax.plot(x, y, label=label, color=color, linewidth=linewidth)
        else:
            plt.plot(x, y, label=label, color=color, linewidth=linewidth)

    def add_scatter(self, x, y, label="", marker="o", color="blue", s=1, zorder=1):
        if self.polar:
            self.ax.scatter(
                x, y, label=label, marker=marker, s=s, color=color, zorder=zorder
            )
        else:
            plt.scatter(
                x, y, label=label, marker=marker, s=s, color=color, zorder=zorder
            )

    def add_errorbar(
        self,
        x,
        y,
        xerror=None,
        yerror=None,
        label="",
        marker="o",
        color="blue",
        s=1,
        capsize=3,
    ):
        plt.errorbar(
            x,
            y,
            xerr=xerror,
            yerr=yerror,
            fmt=marker,
            label=label,
            elinewidth=1,
            markersize=s,
            color=color,
            capsize=capsize,
        )

    def add_vline(self, x, y=0, label="", color="black", linestyle="-", linewidth=2):
        plt.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
        plt.text(x + 1, y, label, rotation=0, verticalalignment="bottom", color=color)

    def add_colormap(
        self,
        Z,
        X,
        Y,
        cbarlabel=r"$\Delta$$S_{21}$\u2009(dB)",
        cmap="hot_r",
        equal_bounds=False,
        vmin=None,
        vmax=None,
    ):
        self.colormap = True

        if X.ndim == 1:
            X, Y = np.meshgrid(np.unique(X), np.unique(Y))

        if vmin is None:
            vmin = np.min(Z)
        if vmax is None:
            vmax = np.max(Z)
        if equal_bounds:
            vmax_abs = np.max(np.abs(Z))
            vmin, vmax = -vmax_abs, vmax_abs

        pcm = plt.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(pcm)
        cbar.ax.tick_params(labelsize=self.fontsize)
        cbar.ax.set_title(cbarlabel, fontsize=self.fontsize)

    def plot(
        self,
        show=True,
        save=False,
        legend=False,
        name="Test1",
        title=None,
        xlabel=None,
        ylabel=None,
        yrotation=90,
        yfontsize=None,
        ymin=None,
        ymax=None,
        xmin=None,
        xmax=None,
        minorticks=False,
        output_folder=OUTPUT_FOLDER,
    ):
        plt.rcParams["font.family"] = "sans-serif"

        if yfontsize is None:
            yfontsize = self.fontsize

        if self.polar:
            self.ax.set_xticklabels(
                [
                    f"\u2001\u2001{xlabel} = 0°",
                    "45°",
                    "90°",
                    "135°",
                    "180°",
                    "225°",
                    "270°",
                    "315°",
                ],
                fontsize=self.fontsize,
            )
            self.ax.set_yticklabels([])
        else:
            if xlabel:
                plt.xlabel(xlabel, fontsize=self.fontsize)
            if ylabel:
                plt.ylabel(ylabel, fontsize=yfontsize, rotation=yrotation)
            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
                width=1,
                length=4,
            )

            if xmin is not None and xmax is not None:
                plt.xlim([xmin, xmax])
            if ymin is not None and ymax is not None:
                plt.ylim([ymin, ymax])

        if legend:
            # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.legend(loc="best")

        if title:
            plt.title(title, loc="left")

        if minorticks:
            plt.minorticks_on()
            plt.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.7)
            plt.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.9)

        if save:
            ext = "png" if self.colormap else "svg"
            prefix = "ColorMap" if self.colormap else ""
            output_filepath = os.path.join(output_folder, f"{prefix}{name}.{ext}")
            plt.savefig(output_filepath, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.clf()

    # Keep old name for backwards compatibility
    plot_Graph = plot


def quick_plot(
    x,
    y,
    xlabel="",
    ylabel="",
    name="Test1",
    title=None,
    output_folder=OUTPUT_FOLDER,
    scatter=True,
    s=1,
    show=True,
    save=False,
    width_cm=16.5,
    height_cm=None,
    color="blue",
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
):
    """Quick single-line plot without using the Graph class."""
    fontsize = 11

    width_in = width_cm / 2.54
    height_cm = height_cm if height_cm else width_cm * (9 / 16)
    height_in = height_cm / 2.54

    plt.figure(figsize=(width_in, height_in), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"

    if title:
        plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tick_params(
        axis="both", direction="in", top=True, right=True, width=1, length=4
    )

    if xmin is not None and xmax is not None:
        plt.xlim([xmin, xmax])
    if ymin is not None and ymax is not None:
        plt.ylim([ymin, ymax])

    if scatter:
        plt.scatter(x, y, label=name, marker="x", s=s, color=color)
    else:
        plt.plot(x, y, label=name, color=color)

    if save:
        output_filepath = os.path.join(output_folder, f"{name}.eps")
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.clf()


# Keep old name for backwards compatibility
GraphPlot = quick_plot


def get_plot_colors(num_colors, alpha=1):
    """Get evenly spaced colors from a rainbow colormap."""
    colors = plt.colormaps["rainbow"](np.linspace(0, 1, num_colors))
    colors[:, 3] = alpha
    return colors
