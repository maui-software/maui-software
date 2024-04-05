"""
	This module offers a streamlined approach to retrieving information on
	audio samples within the Maui project framework. It serves to abstract the
	complexities of accessing and parsing audio file metadata, providing a simple
	method for users to obtain a structured and comprehensive overview of available
	audio samples. The methods returns a pandas DataFrame detailing
	the samples, including aspects such as file paths, durations, and other
	pertinent metadata.

	The functionality leverages the `maui.io` module for the extraction of audio
	information, ensuring consistency and reliability in the data presented.

	Functionality:
	- Simplifies the retrieval of audio sample metadata within the Maui
	  framework.

	Usage:
	- Intended for use in data analysis workflows requiring access to structured
	  information about specific sets of audio samples.

	Dependencies:
	- os: For handling file and directory paths.
	- maui.io: For underlying audio information extraction processes.

	Examples and additional details are provided in the function docstring,
	guiding users in applying the module to their specific needs.
"""

import os
import zipfile

import pandas as pd

import maui.io


def get_audio_sample(dataset: str) -> pd.DataFrame:
    """
    Get Leec Audio Samples available in maui.

    Parameters
    ----------
    dataset : str
        Dataset to be loaded. The available datasets are: leec

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing information about the audio samples.

    Examples
    --------
    To retrieve Leec audio samples and store the information in a DataFrame,
    you can call this function as follows:

    >>> from maui import samples
    >>> df = samples.get_audio_sample(dataset="leec")


    """
    available_datasets = ["leec"]
    if dataset not in available_datasets:
        raise Exception("Dataset not available")

    dataset_format_name = "unknwown"
    if dataset == "leec":
        dataset_format_name = "LEEC_FILE_FORMAT"

    absolute_path = os.path.dirname(__file__)
    relative_path = f"""../data/audio_samples/{dataset}_data.zip"""
    full_path = os.path.join(absolute_path, relative_path)

    # Create a directory to store the extracted files
    os.makedirs(f"""maui_samples_{dataset}""", exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(full_path, "r") as zip_ref:
        # Extract all files to the specified directory
        zip_ref.extractall(f"""maui_samples_{dataset}""")

    df = maui.io.get_audio_info(
        f"""./maui_samples_{dataset}""",
        format_name=dataset_format_name,
        store_duration=1,
        perc_sample=1,
    )

    return df
