    
# maui-software

![maui-softwre logo](maui/data/logo/color_logo_no_background.png "maui-software logo")

**maui-software** is an open-source Python package to visualize eco-acoustic data. This work intends to help ecology specialists to perform data analysis with no need to implement visualizations, enabling faster analysis.

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#active)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2fda82e1f9cd459eb38a78674c544031)](https://app.codacy.com/gh/maui-software/maui-software/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# Operating Systems

`maui-software` was tested on the following operating systems:

  - Windows
  - Ubuntu


# Dependencies

  - python >=3.9,<3.13
  - scikit-maad = 1.3.0
  - plotly >= 5.16.1
  - tqdm >= 4.66.1
  - fpdf >= 1.7.2
  - audioread >= 3.0.0
  - numpy >= 1.19.5
  - pandas >= 1.1.5
  - kaleido = 0.2.1

# Installation

```bash
$ pip install maui-software
```

# Quick Start

To use `maui-software`, one must have a single audio file or, preferencially, a set of audio files and load them as follows:

```python
from maui import io
audio_dir = 'PATH_TO_DIRECTORY'
df = io.get_audio_info(audio_dir, store_duration=1, perc_sample=0.01)

df
```

# Acknowledgements

Special thanks to the spatial ecology and conservation laboratory
(LEEC, in portuguese) at São Paulo State University (UNESP, in portuguese)
for the data set provided and expert support.

Also, `maui-software` uses `scikit-maad` to calculate
acoustic indices and spectrograms from audio. 
We finally acknowledge the financial support of FAPESP
(The State of São Paulo Research Foundation) Grant # 2021/08322-3.
