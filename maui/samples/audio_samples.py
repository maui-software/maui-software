import pandas as pd
import os

import maui.io



def get_leec_audio_sample():
	"""
	Get Leec Audio Samples available in maui.

	Parameters
	----------
	None

	Returns
	-------
	df : pandas.DataFrame
	    A DataFrame containing information about the audio samples.

	Examples
	--------
	To retrieve Leec audio samples and store the information in a DataFrame, you can call this function as follows:

	>>> from maui import samples
	>>> df = samples.get_leec_audio_sample()


	"""

	absolute_path = os.path.dirname(__file__)
	relative_path = '../data/audio_samples/'
	full_path = os.path.join(absolute_path, relative_path)

	df = maui.io.get_audio_info(full_path, store_duration=1, perc_sample=1)

	return df