# -*- coding: utf-8 -*-
"""
The module ``files_metadata`` has a collection of functions to retrieve metadata from the file name


.. autosummary::
    :toctree: generated/

    verify_yaml_format
    get_format_config
    extract_metadata

"""

from .files_metadata import (verify_yaml_format,
                     get_format_config,
                     extract_metadata)


__all__ = [ # files_metadata
           'verify_yaml_format',
           'get_format_config',
           'extract_metadata']