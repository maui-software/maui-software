# -*- coding: utf-8 -*-
"""
The module ``utils`` has a collection of utilities to be used in audio files management.


.. autosummary::
    :toctree: generated/

    
    false_color_spectrogram_prepare_dataset
    segment_audio_files    

"""

from .utils import (false_color_spectrogram_prepare_dataset,
            segment_audio_files)


__all__ = [ # utils
           'false_color_spectrogram_prepare_dataset',
           'segment_audio_files']