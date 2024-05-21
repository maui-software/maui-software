import pytest
import pandas as pd

import maad
from maui import acoustic_indices, samples


def test_calculate_acoustic_indices(tmp_path):
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    # Call the function with a valid dataset
    df = samples.get_audio_sample("leec", tmp_dir)
    df = df.head()

    indices_list = ['median_amplitude_envelope']
    
    result_df = acoustic_indices.calculate_acoustic_indices(df, indices_list, store_df=False, parallel=False)
    # print(result_df)
    
    assert 'm' in result_df.columns
    assert len(result_df) == len(df)

def test_calculate_acoustic_indices_parallel(tmp_path):
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    # Call the function with a valid dataset
    df = samples.get_audio_sample("leec", tmp_dir)
    df = df.head()

    indices_list = ['median_amplitude_envelope']
    
    result_df = acoustic_indices.calculate_acoustic_indices(df, indices_list, store_df=False, parallel=True)
    # print(result_df)
    
    assert 'm' in result_df.columns
    assert len(result_df) == len(df)

