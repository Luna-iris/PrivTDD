"""Portable figure rendering when the publication font is not installed."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib
import matplotlib.pyplot as plt

from privtdd.experiments.render import e2_plot


class PlotFontTests(unittest.TestCase):
    def test_missing_arial_uses_bundled_font_and_renders(self):
        original_findfont = e2_plot.font_manager.findfont

        def without_arial(properties, *args, **kwargs):
            if not properties.get_file() and properties.get_family() == ["Arial"]:
                raise ValueError("Arial unavailable")
            return original_findfont(properties, *args, **kwargs)

        with matplotlib.rc_context(), tempfile.TemporaryDirectory() as directory:
            with patch.object(e2_plot.font_manager, "findfont", side_effect=without_arial):
                with self.assertWarnsRegex(UserWarning, "Arial.*unavailable.*bundled DejaVu Sans"):
                    e2_plot.configure_font("Arial", 8.2)
            self.assertEqual(plt.rcParams["font.sans-serif"], ["DejaVu Sans"])
            figure, axis = plt.subplots()
            try:
                axis.plot([0.5, 1, 2], [4, 2, 1])
                axis.set(xlabel="Privacy budget ε", ylabel="EMD", title="PrivTDD")
                for extension in ("png", "pdf"):
                    path = Path(directory) / f"fallback.{extension}"
                    figure.savefig(path)
                    self.assertGreater(path.stat().st_size, 100)
            finally:
                plt.close(figure)
            legend_figure = plt.figure(figsize=(3.49, .21))
            try:
                e2_plot.add_legend(legend_figure, narrow=True)
                legend_figure.canvas.draw()
                bounds = legend_figure.legends[0].get_window_extent()
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, legend_figure.bbox.width)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.y1, legend_figure.bbox.height)
            finally:
                plt.close(legend_figure)

    def test_available_requested_font_is_retained_without_warning(self):
        with matplotlib.rc_context(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            e2_plot.configure_font("DejaVu Serif", 8.2)
            self.assertEqual(plt.rcParams["font.sans-serif"], ["DejaVu Serif"])
            self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
