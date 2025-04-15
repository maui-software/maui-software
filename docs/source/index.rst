.. maui-software documentation master file, created by
   sphinx-quickstart on Mon Sep 11 20:30:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../maui/data/logo/color_logo_no_background.svg
   :width: 600
   :alt: maui-software logo
   :align: center

Acoustic Ecology Visualization
==============================

**maui-software** maui-software is an open-source Python package to visualize eco-acoustic data. This work intends to help ecology specialists to perform data analysis with no need to implement visualizations, enabling faster analysis.


.. toctree::
   :caption: Documentation
   :maxdepth: 1
   
   maui.acoustic_indices
   maui.eda
   maui.io
   maui.visualizations
   maui.samples
   maui.utils

.. toctree::
   :caption: Example Gallery
   :maxdepth: 1
   
   example_gallery

Operating Systems
=================

``maui-software`` was tested on the following operating systems:

* Windows
* Ubuntu

Dependencies
============

* python = >=3.9,<3.13
* scikit-maad = ^1.3.0
* plotly = ^5.16.1
* tqdm = ^4.66.1
* fpdf = ^1.7.2
* audioread = ^3.0.0
* pandas = >=1.5
* numpy = >= 1.21
* kaleido = 0.2.1
* pyyaml = ^6.0.1
* matplotlib = ^3.8.3
* gdown = ^5.2.0
* pydub = ^0.25.1
* objgraph = ^3.6.1
* pyarrow = ^17.0.0
* fastparquet = ^2024.5.0


Installation
============

``maui-software`` is available in PyPi and cand installed as following::

   pip install maui-software


Quick Start
===========

To use ``maui-software``, one must have a single audio file or, preferencially, a set of audio files and load them as follows::

   from maui import io
   audio_dir = 'PATH_TO_DIRECTORY'
   df = io.get_audio_info(audio_dir, store_duration=1, perc_sample=0.01)
   df

Acknowledgements
================

Special thanks to the spatial ecology and conservation laboratory (LEEC, in portuguese) at São Paulo State University (UNESP, in portuguese) for the data set provided and expert support.

Also, `maui-software` uses `scikit-maad` to calculates acoustic indices and spectrograms from audio. We finally acknowledge the financial support of FAPESP (The State of São Paulo Research Foundation) Grant # 2021/08322-3.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
