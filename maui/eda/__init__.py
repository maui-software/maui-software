# -*- coding: utf-8 -*-
"""

The module ``eda`` has a collection of functions to perform exploratory data analysis.


.. autosummary::
    :toctree: generated/

    card_summary
    landscape_environment_heatmap
    plot_landscape_histogram
    plot_landscape_duration
    plot_landscape_daily_distribution
    plot_environment_histogram
    plot_environment_duration
    plot_environment_daily_distribution
    plot_duration_distribution
    export_file_names_summary_pdf

"""

from .eda import (card_summary,
                     landscape_environment_heatmap,
                     plot_landscape_histogram,
                     plot_landscape_duration,
                     plot_landscape_daily_distribution,
                     plot_environment_histogram,
                     plot_environment_duration,
                     plot_environment_daily_distribution,
                     plot_duration_distribution,
                     export_file_names_summary_pdf)


__all__ = [ # eda
           'card_summary',
           'landscape_environment_heatmap',
           'plot_landscape_histogram',
           'plot_landscape_duration',
           'plot_landscape_daily_distribution',
           'plot_environment_histogram',
           'plot_environment_duration',
           'plot_environment_daily_distribution',
           'plot_duration_distribution',
           'export_file_names_summary_pdf']