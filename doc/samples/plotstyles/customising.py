"""Compare temporary rcParams overrides with the stonerplots base style."""

from pathlib import Path

from cycler import cycler

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xyy")
d.normalise(2, scale=(0.0, 10.0))

with d.plot.style("stoner"):
    figure = d.plot.new_figure(figsize=(7, 8))
    with d.plot.StackVertical(2, figure=figure, joined=False, adjust_figsize=False) as panels:
        d.plot.line(ax=panels[0], title="Style defaults")
        overrides = {
            "axes.prop_cycle": cycler(color=["Blue", "Purple"]) + cycler(linestyle=["-", "--"]),
            "lines.linewidth": 2.0,
        }
        with d.plot.style(overrides):
            # Styles apply at creation; configure an existing axes explicitly.
            ax = panels[1]
            ax.set_prop_cycle(overrides["axes.prop_cycle"])
            d.plot.line(ax=ax, title="Customised settings")
            ax.grid(True, color="green", linestyle="-.")
