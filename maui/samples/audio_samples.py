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
import urllib.request
from pathlib import Path
import requests

import pandas as pd
import gdown
from tqdm import tqdm

import maui.io


def get_dataset_url(
    dataset: str,
) -> str:
    """
    Generate a Google Drive URL for the specified dataset.

    This function returns a direct download URL for the dataset based on the provided dataset name. 
    Currently, it supports the "leec" dataset, for which it constructs a URL from a pre-defined
    Google Drive file ID.

    Parameters
    ----------
    dataset : str
        The name of the dataset for which the download URL is requested.
        Currently supported datasets:
        - "leec": Returns the download URL for the "leec" dataset.

    Returns
    -------
    str
        A string containing the direct download URL from Google Drive for the specified dataset.
        If the dataset is not supported, an empty URL is returned.

    Examples
    --------
    >>> get_dataset_url("leec")
    'https://drive.google.com/uc?id=1tw7BpPNBeS6Dz0XJOwwYuJOYJgd4XSUE'
    
    Notes
    -----
    - This function is designed to handle future datasets by mapping their names to specific Google
      Drive file IDs.
    - If an unsupported dataset is provided, an empty file ID will result in an invalid URL.
    """
    file_id = ""
    if dataset == "leec":
        file_id = "1tw7BpPNBeS6Dz0XJOwwYuJOYJgd4XSUE"
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
    zip_file_path = os.path.join(os.path.dirname(__file__), "data", zip_file_name)
    if extract_path is None:
        extract_path = os.path.join(os.path.dirname(__file__), "data", dataset)

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


def _get_xc_dataset(q: dict) -> pd.DataFrame:
    """
    Retrieves a dataset from the Xeno-canto API based on the provided query parameters.

    Parameters
    ----------
    q : dict
        A dictionary containing the query parameters to filter recordings from the Xeno-canto API.
        Example of keys: 'gen', 'sp', 'cnt', etc.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the recordings data retrieved from the Xeno-canto API.
        If no recordings are found or an error occurs, an empty DataFrame is returned.

    Notes
    -----
    - The method constructs the API URL from the query dictionary and sends a GET request.
    - If the API response contains recordings, they are converted into a DataFrame.
    """

    url = "https://xeno-canto.org/api/2/recordings?query="

    for i, (key, value) in enumerate(q.items()):
        if i >= 1:
            url += " "
        url += key + ":" + value

    response = requests.get(url)

    if response.status_code == 200:
        jsondata = response.json()

        # Check if there are recordings
        if "recordings" in jsondata and jsondata["recordings"]:
            # Create a DataFrame from the recordings data
            df_xc = pd.DataFrame(jsondata["recordings"])
        else:
            print("No recordings found.")
            df_xc = pd.DataFrame()  # Empty DataFrame in case no results
    else:
        print(f"Error: {response.status_code}")
        df_xc = pd.DataFrame()  # Empty DataFrame in case of an error

    return df_xc


def _download_xc_files(df: pd.DataFrame, extract_path: str) -> pd.DataFrame:
    """
    Downloads audio files from Xeno-canto based on a DataFrame of recordings and saves them locally.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the recordings information, including the file URLs and file names.
    extract_path : str
        The directory where the audio files will be saved. If the directory does not exist, it
        is created.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two additional columns:
        - 'local_file_path': The local path where the file was saved (or None if download failed).
        - 'file_downloaded': A boolean indicating whether the file was successfully downloaded.

    Notes
    -----
    - Files that already exist in the destination folder are skipped.
    - If a file download fails, the method will print an error and add a None value to the
    file path.
    """

    extract_path = Path(extract_path)
    os.makedirs(extract_path, exist_ok=True)

    path_list = []
    downloaded_list = []

    # Adding tqdm progress bar to the loop
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading files"):
        file_name_xc = row["file-name"]
        full_path = os.path.join(extract_path, file_name_xc)
        if os.path.exists(full_path):
            print(f"File {file_name_xc} already exists.")
            path_list.append(full_path)
            downloaded_list.append(True)
        else:
            try:
                _, _ = urllib.request.urlretrieve(row["file"], full_path)
                path_list.append(full_path)
                downloaded_list.append(True)
            except (urllib.error.URLError, OSError) as e:
                print(f"File {file_name_xc} could not be downloaded. Error: {str(e)}")
                path_list.append(None)
                downloaded_list.append(False)

    df["local_file_path"] = path_list
    df["file_downloaded"] = downloaded_list

    return df


def get_xc_data(q: dict, extract_path: str) -> pd.DataFrame:
    """
    Retrieves and downloads Xeno-canto data based on a set of query parameters.

    Parameters
    ----------
    q : dict
        A dictionary of query parameters to filter recordings from Xeno-canto.
    extract_path : str
        The directory where the audio files will be saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the recordings data from Xeno-canto, with additional columns for
        the local file paths and file download status.

    Raises
    ------
    ValueError
        If unexpected query parameters are provided.

    Notes
    -----
    - The method first validates the query dictionary to ensure only valid keys are used.
    - After retrieving the recordings data using the `_get_xc_dataset` method, it downloads
      the audio files using the `_download_xc_files` method.

    Example
    --------
    >>> from maui import samples
    >>> params = {
    >>>     'cnt':'brazil'
    >>> }
    >>> df = samples.get_xc_data(q = params, extract_path="./xc_data")
    """

    query_params_list = [
        "id", "gen", "sp", "ssp", "group", "en", "rec", "cnt",
        "loc",   "lat",   "lng",  "type",   "sex", "stage",
        "method", "url", "file", "file-name", "sono", "osci", "lic",
        "q", "length", "time", "date", "uploaded", "also", "rmk",
        "bird-seen", "animal-seen", "playback-used", "temperature",
        "regnr", "auto", "dvc", "mic", "smp",
    ]

    extra_keys = set(q.keys()) - set(query_params_list)

    if extra_keys:
        raise ValueError(f"Unexpected keys found: {extra_keys}")

    df_xc = _get_xc_dataset(q)
    df_xc = _download_xc_files(df_xc, extract_path)

    return df_xc
