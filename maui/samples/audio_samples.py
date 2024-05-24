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
import gdown

import maui.io


def get_dataset_url(dataset: str,) -> str:
    file_id = ""
    if dataset == 'leec':
        file_id = '1tw7BpPNBeS6Dz0XJOwwYuJOYJgd4XSUE'
    return f"https://drive.google.com/uc?id={file_id}"

def get_audio_sample(dataset: str, extract_path: str = None) -> pd.DataFrame:
    """
    Get Leec Audio Samples available in maui.

    Parameters
    ----------
    dataset : str
        Dataset to be loaded. The available datasets are: leec
    extract_path : str
        Directory to extract sample files

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

    dataset_format_name = "unknown"
    if dataset == "leec":
        dataset_format_name = "LEEC_FILE_FORMAT"

    zip_file_name = f"{dataset}.zip"
    zip_file_path = os.path.join(os.path.dirname(__file__), 'data', zip_file_name)
    if extract_path is None:
        extract_path = os.path.join(os.path.dirname(__file__), 'data', dataset)
    
    print(zip_file_path)
    print(extract_path)
    
    file_url = get_dataset_url(dataset)
    
    os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
    gdown.download(file_url, zip_file_path, quiet=False)
    
    # Check if the file is a valid zip file
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.testzip()
            # Extract all files to the specified directory 
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        return None

    df = maui.io.get_audio_info(
        extract_path,
        format_name=dataset_format_name,
        store_duration=1,
        perc_sample=1,
    )

    return df
