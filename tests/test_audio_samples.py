import os
import zipfile
import pandas as pd
import pytest
from maui import samples


def test_get_audio_sample_invalid_dataset():
    # Call the function with an invalid dataset and expect an Exception
    with pytest.raises(Exception):
        samples.get_audio_sample(dataset="invalid_dataset")


def test_get_audio_sample_valid_dataset_leec(tmp_path):
    # Create a temporary directory path
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    # Call the function with a valid dataset
    df = samples.get_audio_sample("leec", tmp_dir)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(os.listdir(tmp_dir)) == 120
