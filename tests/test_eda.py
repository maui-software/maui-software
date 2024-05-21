import pytest
import pandas as pd
import plotly.graph_objs as go
from maui import eda, samples
from maui.eda.eda import CategoryLimitError


# Sample DataFrame for testing
@pytest.fixture
def sample_df():
    data = {
        'file_path': ['file1', 'file2', 'file3', 'file4'],
        'dt': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'duration': [100, 200, 300, 400],
        'landscape': ['urban', 'rural', 'urban', 'rural'],
        'environment': ['indoor', 'outdoor', 'indoor', 'outdoor']
    }
    return pd.DataFrame(data)

def test_card_summary(sample_df):
    # Test with valid categories
    categories = ['landscape', 'environment']
    card_dict, fig = eda.card_summary(sample_df, categories, show_plot=False)
    
    assert isinstance(card_dict, dict)
    assert isinstance(fig, go.Figure)
    assert card_dict['n_samples'] == 4
    assert card_dict['distinct_days'] == 2
    assert card_dict['total_time_duration'] == 1000 / 60
    assert card_dict['mean_time_duration'] == 250 / 60
    assert card_dict['landscape'] == 2
    assert card_dict['environment'] == 2

    with pytest.raises(CategoryLimitError) as excinfo:
        eda.card_summary(sample_df, ['landscape', 'environment', 'extra_category'], show_plot=False)
    assert str(excinfo.value) == "At most two categories should be selected."


def test_heatmap_analysis(sample_df):
    df_group, fig = eda.heatmap_analysis(sample_df, 'landscape', 'environment', show_plot=False)
    
    assert isinstance(df_group, pd.DataFrame)
    assert isinstance(fig, go.Figure)
    assert df_group.shape == (2, 3)
    assert 'landscape' in df_group.columns
    assert 'environment' in df_group.columns
    assert 'count' in df_group.columns

def test_histogram_analysis(sample_df):
    fig = eda.histogram_analysis(sample_df, 'landscape', 'environment', show_plot=False)
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Amount of samples by landscape and segmented by environment'

def test_duration_analysis(sample_df):
    fig = eda.duration_analysis(sample_df, 'landscape', 'duration', show_plot=False)
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Duration distribution by landscape'

def test_daily_distribution_analysis(sample_df):
    fig = eda.daily_distribution_analysis(sample_df, 'dt', 'landscape', show_plot=False)
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Ammount of samples by Day and landscape'

def test_duration_distribution(sample_df):
    fig = eda.duration_distribution(sample_df, show_plot=False)
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Duration distribution'
