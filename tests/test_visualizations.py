import pytest
import pandas as pd
import numpy as np
import wave
from pathlib import Path
import plotly.graph_objects as go
from maui import visualizations

# Sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'index1': [0.1, 0.3, 0.5, 0.7, 0.9],
        'index2': [0.2, 0.4, 0.6, 0.8, 1.0],
        'group': ['A', 'A', 'B', 'B', 'C']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_wav_file(tmp_path):
    file_path = tmp_path / "test.wav"

    # Parameters for the WAV file
    n_channels = 1        # Mono
    sampwidth = 2         # 2 bytes per sample
    framerate = 44100     # Sampling rate in Hz
    n_frames = 44100      # One second of audio

    # Generate a dummy sine wave
    t = np.linspace(0, 1, framerate)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Write the WAV file
    with wave.open(str(file_path), 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(data.tobytes())

    return file_path

# Testing indices_radar_plot function
def test_indices_radar_plot(sample_dataframe):
    fig = visualizations.indices_radar_plot(sample_dataframe, indices=['index1', 'index2'], agg_type='mean', show_plot=False)
    assert isinstance(fig, go.Figure)
    assert 'Radar Plot - Comparisson between indices' in fig.layout.title.text

def test_indices_radar_plot_invalid_agg_type(sample_dataframe):
    with pytest.raises(AssertionError):
        visualizations.indices_radar_plot(sample_dataframe, indices=['index1', 'index2'], agg_type='invalid', show_plot=False)

def test_indices_radar_plot_empty_indices(sample_dataframe):
    with pytest.raises(IndexError):
        visualizations.indices_radar_plot(sample_dataframe, indices=[], agg_type='mean', show_plot=False)

# Testing indices_histogram_plot function
def test_indices_histogram_plot(sample_dataframe):
    fig = visualizations.indices_histogram_plot(sample_dataframe, indices=['index1', 'index2'], show_plot=False)
    assert isinstance(fig, go.Figure)
    assert 'Histogram Plot - Distribution of selected indices' in fig.layout.title.text

def test_indices_histogram_plot_invalid_group_by(sample_dataframe):
    with pytest.raises(AssertionError):
        visualizations.indices_histogram_plot(sample_dataframe, indices=['index1'], group_by='index3', show_plot=False)

def test_indices_histogram_plot_empty_indices(sample_dataframe):
    with pytest.raises(Exception):
        visualizations.indices_histogram_plot(sample_dataframe, indices=[])

# Testing indices_violin_plot function
def test_indices_violin_plot(sample_dataframe):
    fig = visualizations.indices_violin_plot(sample_dataframe, indices=['index1', 'index2'], show_plot=False)
    assert isinstance(fig, go.Figure)
    assert 'Violin Plot - Distribution of selected indices' in fig.layout.title.text

def test_indices_violin_plot_invalid_group_by(sample_dataframe):
    with pytest.raises(AssertionError):
        visualizations.indices_violin_plot(sample_dataframe, indices=['index1'], group_by='invalid', show_plot=False)

def test_indices_violin_plot_empty_indices(sample_dataframe):
    with pytest.raises(AttributeError):
        visualizations.indices_violin_plot(sample_dataframe, indices=[], show_plot=False)

# Testing spectrogram_plot function
def test_spectrogram_plot(sample_wav_file):
    
    fig = visualizations.spectrogram_plot(file_path=str(sample_wav_file), mode='psd', show_plot=False)
    assert isinstance(fig, go.Figure)
    assert 'Spectrogram generated from the file test.wav' in fig.layout.title.text

def test_spectrogram_plot_invalid_mode(sample_wav_file):

    with pytest.raises(AssertionError):
        visualizations.spectrogram_plot(file_path=str(sample_wav_file), mode='invalid', show_plot=False)

# Add more tests as needed for other parameters and edge cases
