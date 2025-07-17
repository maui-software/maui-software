# -*- coding: utf-8 -*-
"""
The module ``visualizations`` has a collection of functions to create visualizations over sudios and its acoustic indices.


.. autosummary::
    :toctree: generated/

    indices_radar_plot
    indices_histogram_plot
    indices_violin_plot
    spectrogram_plot
    false_color_spectrogram_plot
    diel_plot
    polar_bar_plot
    parallel_coordinates_plot

"""

from .visualizations import (indices_radar_plot,
                     indices_histogram_plot,
                     indices_violin_plot,
                     spectrogram_plot,
                     false_color_spectrogram_plot,
                     diel_plot,
                     polar_bar_plot,
                     parallel_coordinates_plot)


__all__ = [ # visualizations
           'indices_radar_plot',
           'indices_histogram_plot',
           'indices_violin_plot',
           'spectrogram_plot',
           'false_color_spectrogram_plot',
           'diel_plot',
           'polar_bar_plot',
           'parallel_coordinates_plot']
