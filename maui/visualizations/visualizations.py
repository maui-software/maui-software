"""
    This module offers visualization tools for acoustic analysis, providing
    functions to generate radar, histogram, violin, and spectrogram plots. These
    visualizations aid in the comparison and analysis of acoustic indices
    extracted from audio files, facilitating the understanding of soundscapes
    and audio properties. The module leverages Plotly for generating interactive
    plots, offering flexibility in exploring and presenting acoustic data.

    Key functionalities include:
    - Radar plots for comparing multiple indices across different categories or
      groups.
    - Histogram plots to visualize the distribution of indices.
    - Violin plots for detailed distribution analysis of indices, showing density
      and distribution shape.
    - Spectrogram plots for visualizing the frequency content of audio files over
      time.

    These tools are designed for researchers, ecologists, and sound engineers
    interested in analyzing audio data, particularly for environmental sound
    analysis, bioacoustics, and similar fields.

    Functions:
    - indices_radar_plot: Generates radar plots for comparing acoustic indices.
    - indices_histogram_plot: Creates histogram plots to visualize index
      distributions.
    - indices_violin_plot: Produces violin plots for detailed distribution analysis.
    - spectrogram_plot: Computes and visualizes spectrograms of audio files.

    Usage examples and parameters for each function are provided within their
    respective docstrings, guiding their application in various analysis scenarios.

    Dependencies:
    - pandas for data manipulation.
    - plotly for interactive plotting.
    - maad for acoustic feature extraction and analysis.

    Note:
    - Ensure that audio files are accessible and properly formatted before analysis.
    - Function parameters allow customization of plots, including aspects like
      figure size and grouping.
"""

import math
import copy
import warnings
import os
import re

from dateutil import parser
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from maad import sound, util


def indices_radar_plot(
    df,
    indices: list,
    agg_type: str,
    group_by: list = None,
    max_cols: int = 3,
    fig_size: dict = None,
    show_plot: bool = True,
):
    """
    Create a radar plot to compare indices in a DataFrame.

    This function generates a radar plot to compare multiple indices from a DataFrame.
    It allows aggregating data based on specified aggregation types and grouping by
    one or two columns from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    indices : list
        A list of column names in the DataFrame representing the indices to be compared.
    agg_type : str
        The type of aggregation to be applied ('mean', 'median', 'stddev', 'var', 'max', 'min').
    group_by : list, optional
        A list of one or two column names for grouping data (default is None).
    max_cols : int, optional
        Maximum number of columns for subplots (default is 3).
    fig_size : dict, optional
        A dictionary specifying the height and width of the plot (default is None).
    show_plot : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the radar plot.

    Raises
    ------
    AssertionError
        If the arguments are not correctly specified.
    Exception
        If the input data or arguments are invalid.

    Examples
    --------
    >>> from maui import samples, acoustic_indices, visualizations
    >>> df = samples.get_leec_audio_sample()
    >>> indices_list = ['median_amplitude_envelope', 'temporal_entropy']
    >>> df = acoustic_indices.calculate_acoustic_indices(df, indices_list, parallel=False)
    >>> fig = visualizations.indices_radar_plot(df, indices=['m', 'ht'],
            agg_type='mean', group_by=['environment'], max_cols=3)
    # Generates a radar plot comparing 'Index1' and 'Index2' aggregated by 'Category'.

    Notes
    -----
    - The 'agg_type' argument must be one of ['mean', 'median', 'stddev', 'var', 'max', 'min'].
    - The 'group_by' argument can contain one or two columns for grouping data.
    - 'fig_size' should be a dictionary with 'height' and 'width' keys.
    """

    # 0. Initial configuration
    # 0.1. Verify if agg_type is available
    agg_options = ["mean", "median", "stddev", "var", "max", "min"]
    assert agg_type in agg_options, f"'{agg_type}' is not in {agg_options}"

    # 0.2. Verify if group_by column is available
    if group_by is not None:
        for col in group_by:
            assert col in df.columns, f"'{col}' is not in {df.column}"

        # 0.2.1. Verify if there is a maximum of two categories to group by
        if len(group_by) > 2:
            raise AttributeError("Sorry, the maximum categories to group by is 2")

    # 0.3. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0:
        raise IndexError("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, (
            f"'{index}' is not in {df.columns}. "
            "Verify if it is correctly spelled and if it have been calculated already."
        )

    # 0.4. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if "height" not in fig_size.keys() or "width" not in fig_size.keys():
            raise AttributeError("fig_size must contain width and height keys.")

    # 0.5 Create a copy of the dataframe
    df_unpivot = copy.deepcopy(df)

    # --------------------------------------------

    # 1. Normalize columns to better suit the plot
    for index in indices:
        df_unpivot[index] = (df_unpivot[index] - df_unpivot[index].min()) / (
            df_unpivot[index].max() - df_unpivot[index].min()
        )

    # --------------------------------------------

    # 2. Unpivot dataframe
    df_unpivot = pd.melt(df_unpivot, id_vars=group_by, value_vars=indices)

    # --------------------------------------------

    # 3. Aggregate data
    gb_cols = ["variable"]
    if group_by is not None:
        for col in group_by:
            gb_cols.append(col)

    if agg_type == "mean":
        df_unpivot = df_unpivot.groupby(gb_cols).mean().reset_index()
    if agg_type == "median":
        df_unpivot = df_unpivot.groupby(gb_cols).median().reset_index()
    if agg_type == "stddev":
        df_unpivot = df_unpivot.groupby(gb_cols).std().reset_index()
    if agg_type == "var":
        df_unpivot = df_unpivot.groupby(gb_cols).var().reset_index()
    if agg_type == "max":
        df_unpivot = df_unpivot.groupby(gb_cols).max().reset_index()
    if agg_type == "min":
        df_unpivot = df_unpivot.groupby(gb_cols).min().reset_index()

    df_unpivot = df_unpivot.rename(columns={"variable": "index"})

    # --------------------------------------------

    # 4. Plot data

    n_cols = 1
    n_rows = 1

    if group_by is not None and len(group_by) > 1:
        n_cols = min(len(list(df[group_by[0]].unique())), max_cols)
        n_rows = math.ceil(len(list(df[group_by[0]].unique())) / max_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "polar"}] * n_cols] * n_rows,
        subplot_titles=[" "] * n_cols * n_rows,
    )

    col = 1
    row = 1
    showlegend = True

    colors = px.colors.qualitative.Plotly

    # se não for agrupar
    if group_by is None:
        df_tmp = copy.copy(df_unpivot)

        r = list(df_tmp["value"])
        r.append(r[0])
        theta = list(df_tmp["index"])
        theta.append(theta[0])

        fig.add_trace(go.Scatterpolar(r=r, theta=theta, mode="lines"), row=1, col=1)
    else:
        if len(group_by) == 1:
            lables_list = list(df[group_by[0]].unique())
        else:
            lables_list = list(df[group_by[1]].unique())
        if len(lables_list) > len(colors):
            warnings.warn(
                "There are more categories than available color, "
                "some categories may use the same color"
            )

        for ind, category in enumerate(list(df[group_by[0]].unique())):

            df_tmp = copy.copy(df_unpivot[df_unpivot[group_by[0]] == category])

            if len(group_by) == 1:
                r = list(df_tmp["value"])
                r.append(r[0])
                theta = list(df_tmp["index"])
                theta.append(theta[0])

                fig.add_trace(
                    go.Scatterpolar(
                        name=category,
                        r=r,
                        theta=theta,
                        mode="lines",
                        legendgroup=category,
                        showlegend=showlegend,
                    ),
                    row=row,
                    col=col,
                )
                if category in lables_list:
                    lables_list.remove(category)
                if len(lables_list) == 0:
                    showlegend = False

            else:

                for j, filter_col in enumerate(list(df[group_by[1]].unique())):

                    df_tmp_final = copy.copy(df_tmp[df_tmp[group_by[1]] == filter_col])

                    r = list(df_tmp_final["value"])
                    if len(r) > 0:
                        r.append(r[0])
                        theta = list(df_tmp_final["index"])
                        theta.append(theta[0])

                        if filter_col not in lables_list:
                            showlegend = False
                        else:
                            lables_list.remove(filter_col)
                            showlegend = True

                        fig.add_trace(
                            go.Scatterpolar(
                                name=filter_col,
                                r=r,
                                theta=theta,
                                mode="lines",
                                legendgroup=filter_col,
                                showlegend=showlegend,
                                line_color=colors[j % len(colors)],
                            ),
                            row=row,
                            col=col,
                        )
                        fig.update_polars(radialaxis_showticklabels=False)

                fig.layout.annotations[ind]["text"] = category
                fig.layout.annotations[ind]["yshift"] = 25

                if col >= max_cols:
                    col = 1
                    row += 1
                else:
                    col += 1

    fig.update_layout(title="Radar Plot - Comparisson between indices", title_x=0.5)

    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size["height"], width=fig_size["width"])
    fig.update_layout(polar={"radialaxis": {"showticklabels": False}})

    if show_plot:
        fig.show()
    return fig


# -----------------------------------------------------------------------


def indices_histogram_plot(
    df,
    indices: list,
    group_by: str = None,
    max_cols: int = 3,
    fig_size: dict = None,
    show_plot: bool = True,
):
    """
    Create histogram plots to visualize the distribution of indices in a DataFrame.

    This function generates histogram plots to visualize the distribution of one
    or more indices from a DataFrame. It provides the option to group data by a
    single category column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    indices : list
        A list of column names in the DataFrame representing the indices to be plotted.
    group_by : str, optional
        A column name for grouping data (default is None).
    max_cols : int, optional
        Maximum number of columns for subplots (default is 3).
    fig_size : dict, optional
        A dictionary specifying the height and width of the plot (default is None).
    show_plot : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the histogram plot.

    Raises
    ------
    AssertionError
        If the arguments are not correctly specified.
    Exception
        If the input data or arguments are invalid.

    Examples
    --------
    >>> from maui import samples, acoustic_indices, visualizations
    >>> df = samples.get_leec_audio_sample()
    >>> indices_list = ['median_amplitude_envelope', 'temporal_entropy']
    >>> df = acoustic_indices.calculate_acoustic_indices(df, indices_list, parallel=False)
    >>> fig = visualizations.indices_histogram_plot(df, indices=['m', 'ht'],
    group_by=None, max_cols=3)

    Notes
    -----
    - The 'group_by' argument is optional, but if provided, only one index can be plotted.
    - 'fig_size' should be a dictionary with 'height' and 'width' keys.
    """

    # 0. Initial configuration
    # 0.1. Verify if group_by column is available

    if group_by is not None:
        assert group_by in list(
            df.columns
        ), f"'{group_by}' is not in {list(df.columns)}"
        if len(indices) > 1:
            raise Exception(
                "Sorry, to group by some category, only one index is supported."
            )

    # 0.2. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0:
        raise Exception("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, (
            f"'{index}' is not in {df.columns}. "
            "Verify if it is correctly spelled and if it have been calculated already."
        )

    # 0.3. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if "height" not in fig_size.keys() or "width" not in fig_size.keys():
            raise Exception("fig_size must contain width and height keys.")

    # --------------------------------------------

    # 1. Plot data

    if group_by is None:
        n_cols = min(len(indices), max_cols)
        n_rows = math.ceil(len(indices) / max_cols)
    else:
        n_cols = min(len(df[group_by].unique()), max_cols)
        n_rows = math.ceil(len(df[group_by].unique()) / max_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "histogram"}] * n_cols] * n_rows,
        subplot_titles=[" "] * n_cols * n_rows,
    )

    col = 1
    row = 1
    showlegend = True

    # add one trace for each index
    if group_by is None:
        for i, index in enumerate(indices):

            fig.add_trace(
                go.Histogram(name=index, x=df[index], showlegend=showlegend),
                row=row,
                col=col,
            )
            fig.layout.annotations[i]["text"] = index
            fig.layout.annotations[i]["yshift"] = 25

            if col >= max_cols:
                col = 1
                row += 1
            else:
                col += 1
    else:
        for i, category in enumerate(df[group_by].unique()):
            df_index = df[df[group_by] == category]

            fig.add_trace(
                go.Histogram(name=index, x=df_index[indices[0]], showlegend=showlegend),
                row=row,
                col=col,
            )
            fig.layout.annotations[i]["text"] = category
            fig.layout.annotations[i]["yshift"] = 25

            if col >= max_cols:
                col = 1
                row += 1
            else:
                col += 1

    fig.update_layout(
        title="Histogram Plot - Distribution of selected indices", title_x=0.5
    )

    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size["height"], width=fig_size["width"])
    fig.update_layout(polar={"radialaxis": {"showticklabels": False}})

    if show_plot:
        fig.show()
    return fig


# -----------------------------------------------------------------------


def indices_violin_plot(
    df,
    indices: list,
    group_by: str = None,
    fig_size: dict = None,
    show_plot: bool = True,
):
    """
    Create violin plots to visualize the distribution of indices in a DataFrame.

    This function generates violin plots to visualize the distribution of one or
    more indices from a DataFrame. It provides the option to group data by a single
    category column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    indices : list
        A list of column names in the DataFrame representing the indices to be plotted.
    group_by : str, optional
        A column name for grouping data (default is None).
    fig_size : dict, optional
        A dictionary specifying the height and width of the plot (default is None).
    show_plot : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the violin plot.

    Raises
    ------
    AssertionError
        If the arguments are not correctly specified.
    Exception
        If the input data or arguments are invalid.

    Examples
    --------
    >>> from maui import samples, acoustic_indices, visualizations
    >>> df = samples.get_leec_audio_sample()
    >>> indices_list = ['median_amplitude_envelope', 'temporal_entropy']
    >>> df = acoustic_indices.calculate_acoustic_indices(df, indices_list, parallel=False)
    >>> fig = visualizations.indices_violin_plot(df, indices=['m', 'ht'], group_by=None)

    Notes
    -----
    - The 'group_by' argument is optional and allows grouping data by a single category column.
    - 'fig_size' should be a dictionary with 'height' and 'width' keys.
    """

    # 0. Initial configuration
    # 0.1. Verify if group_by column is available
    if group_by is not None:
        assert group_by in df.columns, f"'{group_by}' is not in {df.columns}"

    # 0.2. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0:
        raise AttributeError("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, (
            f"'{index}' is not in {df.columns}. "
            "Verify if it is correctly spelled and if it have been calculated already."
        )

    # 0.4. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if "height" not in fig_size.keys() or "width" not in fig_size.keys():
            raise AttributeError("fig_size must contain width and height keys.")

    # --------------------------------------------
    # 2. Plot data

    n_cols = 1
    n_rows = len(indices)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "histogram"}] * n_cols] * n_rows,
        subplot_titles=[" "] * n_cols * n_rows,
    )

    showlegend = True

    violin_width = 0
    if fig_size is not None and group_by is not None:
        violin_width = (
            fig_size["width"] / (len(list(df[group_by].unique())))
        ) / fig_size["width"]
    elif fig_size is not None and group_by is None:
        violin_width = 0.3

    colors = px.colors.qualitative.Plotly

    if group_by is not None:
        lables_list = list(df[group_by].unique())
        categories = list(df[group_by].unique())

        if len(lables_list) > len(colors):
            warnings.warn(
                "There are more categories than available color, "
                "some categories may use the same color"
            )

    # add one trace for each index
    for i, index in enumerate(indices):

        if group_by is not None:

            for j, lab in enumerate(categories):

                fig.add_trace(
                    go.Violin(
                        x=df[group_by][df[group_by] == lab],
                        y=df[index][df[group_by] == lab],
                        name=lab,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        scalemode="width",
                        width=violin_width,
                        legendgroup=lab,
                        showlegend=showlegend,
                        marker_color=colors[j % len(colors)],
                        line_color=colors[j % len(colors)],
                    ),
                    row=i + 1,
                    col=1,
                )

                if lab in lables_list:
                    lables_list.remove(lab)
                if len(lables_list) == 0:
                    showlegend = False
        else:
            fig.add_trace(
                go.Violin(
                    y=df[index],
                    name="",
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    scalemode="width",
                    width=violin_width,
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )

        fig.layout.annotations[i]["text"] = index
        fig.layout.annotations[i]["yshift"] = 25

    fig.update_layout(
        title="""Violin Plot - Distribution of selected indices""", title_x=0.5
    )

    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size["height"], width=fig_size["width"])

    if show_plot:
        fig.show()
    return fig


# -----------------------------------------------------------------------


def spectrogram_plot(
    file_path: str,
    mode: str = None,
    window: str = "hann",
    nperseg: int = 1024,
    noverlap: int = None,
    verbose: bool = False,
    fig_size: dict = None,
    show_plot: bool = True,
):
    """
    Create a spectrogram plot from an audio file.

    This function loads an audio file, computes its spectrogram, and generates a
    heatmap plot to visualize the frequency content over time.

    Parameters
    ----------
    file_path : str
        The path to the audio file.
    mode : str, optional
        The spectrogram mode ('psd', 'mean', 'complex'). Default is None.
    window : str, optional
        The window function to be used for the spectrogram calculation. Default is 'hann'.
    nperseg : int, optional
        The number of data points used in each block for the FFT. Default is 1024.
    noverlap : int, optional
        The number of points of overlap between blocks. Default is None.
    verbose : bool, optional
        Whether to display verbose information during computation. Default is False.
    fig_size : dict, optional
        A dictionary specifying the height and width of the plot. Default is None.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object representing the spectrogram plot.

    Raises
    ------
    AssertionError
        If the arguments are not correctly specified.
    Exception
        If there are errors in the file loading or spectrogram computation.

    Examples
    --------
    >>> from maui import samples, visualizations
    >>> df = samples.get_leec_audio_sample()
    >>> file_path = df.at[df.index[1],'file_path']
    >>> mode = 'psd'
    >>> fig = visualizations.spectrogram_plot(file_path, mode=mode)

    Notes
    -----
    - The 'mode' parameter specifies the type of spectrogram to be computed:
      Power Spectral Density ('psd'), Amplitude Spectrogram ('amplitude'),
      or Complex Spectrogram ('complex').
    - 'fig_size' should be a dictionary with 'height' and 'width' keys.
    """

    # 0. Validations
    # 0.1. available modes
    mode_options = ["psd", "mean", "complex"]
    assert mode in mode_options, f"'{mode}' is not in {mode_options}"

    # 0.2. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if "height" not in fig_size.keys() or "width" not in fig_size.keys():
            raise AttributeError("fig_size must contain width and height keys.")

    # --------------------------------------------

    # 1. Load file
    s, fs = sound.load(file_path)

    # 2. Calculate spectrogram
    sxx, tn, fn, _ = sound.spectrogram(
        s,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
        verbose=verbose,
        mode=mode,
        window=window,
    )

    if mode == "psd":
        sxx_disp = util.power2dB(sxx)
    if mode == "amplitude":
        sxx_disp = util.amplitude2dB(sxx)
    if mode == "complex":
        sxx_disp = util.amplitude2dB(sxx)

    fig_size = {"height": 500, "width": 1200}

    fig = go.Figure(
        data=go.Heatmap(z=sxx_disp, x=tn, y=fn, colorscale="gray", hoverinfo=None)
    )

    fig.update_layout(
        title=f"""Spectrogram generated from the file {os.path.basename(file_path)}""",
        title_x=0.5,
    )

    if fig_size is not None:
        fig.update_layout(height=fig_size["height"], width=fig_size["width"])

    if show_plot:
        fig.show()
    return fig


# -----------------------------------------------------------------------


def _parse_time_format(time_str: str):
    """
    Parse a time string into a 24-hour format (HH:MM).

    Parameters
    ----------
    time_str : str
        The time string to parse. Can be in various formats (e.g., '9am', '09:00').

    Returns
    -------
    str
        The parsed time in HH:MM format.

    Raises
    ------
    ValueError
        If the input time string is invalid or cannot be parsed.
    """
    try:

        # Use dateutil's parser to interpret different time formats
        parsed_time = parser.parse(time_str)
        # Format the time in HH:MM (24-hour format)
        return parsed_time.strftime("%H:%M")
    except (ValueError, TypeError) as exc:
        # Raise an exception if the format can't be parsed
        raise ValueError(f"Invalid time format: {time_str}") from exc


def _truncate_time_to_bin(time_str: str, time_bin_size: int) -> str:
    """
    Truncate the given time string to the nearest time bin based on `time_bin_size`.

    Parameters
    ----------
    time_str : str
        The time string in HH:MM format.
    time_bin_size : int
        The size of the time bin in minutes (e.g., 5, 10, 15).

    Returns
    -------
    str
        The truncated time string in HH:MM format.
    """
    # Parse the time string to a datetime object
    parsed_time = pd.to_datetime(time_str, format="%H:%M")

    # Extract hour and minute
    minute = parsed_time.minute

    # Truncate minute to the nearest bin
    truncated_minute = (minute // time_bin_size) * time_bin_size

    # Replace the minute in the parsed time
    truncated_time = parsed_time.replace(minute=truncated_minute)

    # Return the formatted truncated time in HH:MM format
    return truncated_time.strftime("%H:%M")


def _aggregate_dataframe(
    df: pd.DataFrame, gb_cols: list, grouped_col: str, agg_type: str
) -> pd.DataFrame:
    """
    Aggregate a DataFrame by specified columns and aggregation type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to aggregate.
    gb_cols : list
        List of columns to group by.
    grouped_col : str
        The column to aggregate.
    agg_type : str
        The type of aggregation to perform. Options: "mean", "median",
        "stddev", "var", "max", "min".

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.

    Raises
    ------
    AttributeError
        If the `agg_type` is not one of the supported options.
    """

    if agg_type == "mean":
        df_agg = df.groupby(gb_cols).mean(grouped_col).reset_index()
    elif agg_type == "median":
        df_agg = df.groupby(gb_cols).median(grouped_col).reset_index()
    elif agg_type == "stddev":
        df_agg = df.groupby(gb_cols).std(grouped_col).reset_index()
    elif agg_type == "var":
        df_agg = df.groupby(gb_cols).var(grouped_col).reset_index()
    elif agg_type == "max":
        df_agg = df.groupby(gb_cols).max(grouped_col).reset_index()
    elif agg_type == "min":
        df_agg = df.groupby(gb_cols).min(grouped_col).reset_index()
    else:
        agg_options = ["mean", "median", "stddev", "var", "max", "min"]
        raise AttributeError(f"'{agg_type}' is not in {agg_options}.")

    df_agg = df_agg.rename(columns={grouped_col: "metric"})

    return df_agg


def diel_plot(
    df: pd.DataFrame,
    date_col: str,
    time_col: str,
    duration_col: str,
    time_bin_size: int,
    color_map_col: str,
    agg_type: str = None,
    show_plot: bool = True,
    **kwargs,
):
    """
    Create a diel plot (heatmap) based on time and date columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing date, time, and color mapping columns.
    date_col : str
        Column name for the date in the DataFrame.
    time_col : str
        Column name for the time in the DataFrame.
    duration_col : str
        Column name for the duration of each event in the DataFrame.
    time_bin_size : int
        The size of the time bin in minutes. Must be between 1 and 60.
    color_map_col : str
        Column used to color the plot. Can be numeric or categorical.
    agg_type : str, optional
        Aggregation type for numeric `color_map_col`. Default is None.
    show_plot : bool, optional
        Whether to show the plot. Default is True.
    **kwargs : dict
        Additional arguments for plot customization, such as `height` and `width`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated diel plot as a Plotly figure.

    Raises
    ------
    AttributeError
        If the `time_bin_size` is not between 1 and 60, or if `color_map_col` is
        not of numeric or string type.

    Warnings
    --------
    UserWarning
        If any rows have durations greater than the `time_bin_size`, or if the
        date column contains invalid dates.

    Examples
    --------
    >>> from maui import samples, utils, visualizations
    >>> df = samples.get_audio_sample(dataset="leec")
    >>> def convert_to_seconds(duration_str):
    >>> try:
    >>>     minutes, seconds = map(int, duration_str.split(':'))
    >>>     return minutes * 60 + seconds
    >>> except ValueError:
    >>>     # Handle the case where the input is not in "mm:ss" format
    >>>     raise ValueError(f"Invalid duration format: {duration_str}")
    >>>
    >>> # Apply the function to the 'duration' column
    >>> df = pd.read_csv('xc_data.csv')
    >>> df['length'] = df['length'].apply(convert_to_seconds)
    >>>
    >>> df = df[~df['time'].str.contains(r'?', na=False)]
    >>> df = df[df['time'] != 'am']
    >>> df = df[df['time'] != 'pm']
    >>> df = df[df['time'] != 'xx:xx']
    >>> df = df[df['time'] != '?:?']
    >>> fig = visualizations.diel_plot(df, date_col='date',
    >>>                                 time_col='time', duration_col='length',
    >>>                                 time_bin_size=1, color_map_col='group',
    >>>                                 show_plot= True)

    """

    # 0. Prepare date and time columns
    # 0.1. Parse time column

    df[time_col] = df[time_col].apply(_parse_time_format)

    # 0.2. Verify overlaps
    if time_bin_size < 1 or time_bin_size > 60:
        raise AttributeError(
            "time_bin_size must be an integer between 1 and 60, representing minutes"
        )
    df_time_check = df[df[duration_col] > time_bin_size*60]

    if len(df_time_check) > 0:
        warnings.warn(
            f"Warning: {len(df_time_check)} rows have a duration greater than the "
            f"time_bin_size of {time_bin_size} minutes. The time will be "
            "truncated according to time_bin_size. You should consider "
            "segmenting audio files so each one does not exceed "
            "time_bin_size duration."
        )

    # 0.3. Truncate time column according to time_bin_size
    df[time_col] = df[time_col].apply(lambda t: _truncate_time_to_bin(t, time_bin_size))

    # 0.4. Force date format
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        warnings.warn(
            f"Warning: {invalid_dates} rows have invalid dates. This rows "
            "will be ignored in the visualization."
        )
    df = df.dropna(subset=[date_col])

    # 1. Aggregate dataframe
    if pd.api.types.is_string_dtype(df[color_map_col]):
        # count by color_map_col
        df_plot = df.groupby([date_col, time_col]).size().reset_index(name="metric")
        color_title = "Number of samples"
    elif pd.api.types.is_numeric_dtype(df[color_map_col]):
        # aggregate color_map_col
        if agg_type is None:
            raise AttributeError("agg_type should not be None")
        df_plot = _aggregate_dataframe(
            df, [date_col, time_col], color_map_col, agg_type
        )
        color_title = f"{agg_type} of {color_map_col}"
    else:
        raise AttributeError(f"'{color_map_col}' should be string or numeric type.")

    #  2. Plot image
    fig = go.Figure(
        data=go.Heatmap(
            z=df_plot["metric"],
            x=df_plot[time_col],
            y=df_plot[date_col],
            colorscale="Viridis",
            colorbar={"title": color_title},
        )
    )
    # Set axis labels
    fig.update_layout(
        title="Diel Plot", title_x=0.5, xaxis_title="Time of day", yaxis_title="Date"
    )
    if "height" in kwargs.keys() and "width" in kwargs.keys():
        fig.update_layout(height=kwargs["height"], width=kwargs["width"])

    if show_plot:
        fig.show()

    return fig


# -----------------------------------------------------------------------


def _display_false_color_spectrogram(
    df: pd.DataFrame,
    fc_spectrogram: np.array,
    indices: list,
    fig_size: dict,
    tick_interval: int,
):
    """
    Display a false color spectrogram using Plotly.

    This function visualizes a false color spectrogram generated from
    acoustic indices. The spectrogram is displayed using Plotly with
    customized hover text and axis formatting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the timestamps for the spectrogram.

    fc_spectrogram : np.array
        A 3D numpy array representing the false color spectrogram,
        where the third dimension corresponds to the color channels (R, G, B).

    fig_size : dict
        Dictionary specifying the figure size with 'width' and 'height' keys.
        If None, default values {'width': 2000, 'height': 1000} are used.

    tick_interval : int
        Interval for selecting ticks on the x-axis. If None, the default value is 40.

    Raises
    ------
    AttributeError
        If `fig_size` does not contain both 'width' and 'height' keys.

    Notes
    -----
    - The spectrogram is displayed with customized hover text showing the timestamp
      for each pixel.
    - The function uses Plotly's `go.Figure` and `go.Image` for rendering the image.
    - The layout is updated to ensure the spectrogram is displayed correctly
      with proper scaling and formatting.
    """

    fig_size = {"width": 2000, "height": 1000} if fig_size is None else fig_size
    tick_interval = 40 if tick_interval is None else tick_interval
    if "height" not in fig_size.keys() or "width" not in fig_size.keys():
        raise AttributeError("fig_size must contain width and height keys.")

    # 3.1 Create the figure
    fig = go.Figure()

    # 3.2. Add the image trace with hover text
    hover_text = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    # Create hover text for each pixel in the image
    customdata = np.array([hover_text] * fc_spectrogram.shape[0])

    fig.add_trace(
        go.Image(
            z=fc_spectrogram,
            customdata=customdata,
            hovertemplate="Timestamp: %{customdata}<extra></extra>",
        )
    )

    width = None
    height = None
    if fig_size is not None:
        width = fig_size["width"]
        height = fig_size["height"]

    # Create the x-axis values based on the timestamp
    x_axis_values = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    # Select a subset of x ticks based on the tick_interval
    tick_indices = list(range(0, len(x_axis_values), tick_interval))
    tick_values = [x_axis_values[i] for i in tick_indices]

    # 3.3. Update layout for better visualization
    fig.update_layout(
        title=f"""{re.sub(r'_per_bin', '', indices[0])} (R), """
        f"""{re.sub(r'_per_bin', '', indices[0])} (G) and {indices[2]} """
        f"""(B) False Color Spectrogram""",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "tickvals": tick_indices,
            "ticktext": tick_values,
            "tickangle": 90,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "autorange": True,
            "range": [0, fc_spectrogram.shape[0]],
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        width=width,
        height=height,
    )

    # Display the image
    fig.show()


def false_color_spectrogram_plot(
    df,
    datetime_col: str,
    indices: list,
    display: bool = True,
    unit: str = "scale_60",
    **kwargs,
) -> np.array:
    """
    Generate and optionally display a false color spectrogram from acoustic indices.

    This function creates a false color spectrogram by normalizing and combining
    selected acoustic indices from a DataFrame. The spectrogram can be displayed
    using Plotly and is returned as a 3D numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the acoustic indices and timestamp data.

    datetime_col : str
        Name of the column in `df` that contains datetime values.

    indices : list
        List of column names corresponding to the acoustic indices to be used
        for the R, G, and B channels of the spectrogram.

    display : bool, optional
        If True, the spectrogram is displayed using Plotly. Default is True.

    unit : str, optional
        The time unit to truncate the timestamps from 0.2 seconds to 60 seconds.
        Must be one of ['scale_02', 'scale_04', 'scale_06', 'scale_2', 'scale_4',
        'scale_6', 'scale_12', 'scale_24']. Default is 'scale_60'.

    **kwargs : dict, optional
        Additional arguments for customizing the display:

        - fig_size (dict): Dictionary specifying the figure size with 'width'
          and 'height' keys.
        - tick_interval (int): Interval for selecting ticks on the x-axis.

    Returns
    -------
    np.array
        A 3D numpy array representing the false color spectrogram,
        where the third dimension corresponds to the color channels (R, G, B).

    Raises
    ------
    IndexError
        If `indices` is None or empty.

    AssertionError
        If any of the specified `indices` are not found in the DataFrame columns.

    Exception
        If `unit` is not one of the available units.

    Notes
    -----
    - The function first checks that the selected indices are available in the DataFrame
      and that the specified time unit is valid.
    - The DataFrame is sorted by timestamp, and timestamps are truncated according
      to the specified unit.
    - Acoustic indices are normalized to the range [0, 255] and combined to form
      the false color spectrogram.
    - If `display` is True, the spectrogram is displayed using Plotly,
      with customizable figure size and tick interval.

    Examples
    --------
    >>> from maui import samples, utils, visualizations
    >>> df = samples.get_audio_sample(dataset="leec")
    >>> df["dt"] = pd.to_datetime(df["timestamp_init"]).dt.date
    >>> def pre_calculation_method(s, fs):
    >>>     Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs)
    >>>     Sxx_noNoise= maad.sound.median_equalizer(Sxx_power, display=False, extent=ext)
    >>>     Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>>
    >>>     Sxx, tn, fn, ext = maad.sound.spectrogram(s, fs, mode='amplitude')
    >>>
    >>>     pre_calc_vars = {'Sxx': Sxx, 'tn':tn , 'fn':fn , 'ext':ext,
    >>>                      'Sxx_dB_noNoise':Sxx_dB_noNoise }
    >>>     return pre_calc_vars
    >>>
    >>> def get_aci(pre_calc_vars):
    >>>     aci_xx, aci_per_bin, aci_sum = (
    >>>             maad.features.acoustic_complexity_index(pre_calc_vars['Sxx']))
    >>>     indices = {'aci_xx': aci_xx, 'aci_per_bin':aci_per_bin , 'aci_sum':aci_sum}
    >>>     return indices
    >>>
    >>> def get_spectral_events(pre_calc_vars):
    >>>     EVNspFract_per_bin, EVNspMean_per_bin, EVNspCount_per_bin, EVNsp = (
    >>>             maad.features.spectral_events(
    >>>                 pre_calc_vars['Sxx_dB_noNoise'],
    >>>                 dt=pre_calc_vars['tn'][1] - pre_calc_vars['tn'][0],
    >>>                 dB_threshold=6,
    >>>                 rejectDuration=0.1,
    >>>                 display=False,
    >>>                 extent=pre_calc_vars['ext'])
    >>>             )
    >>>
    >>>     indices = {'EVNspFract_per_bin': EVNspFract_per_bin,
    >>>                'EVNspMean_per_bin':EVNspMean_per_bin,
    >>>                'EVNspCount_per_bin':EVNspCount_per_bin, 'EVNsp':EVNsp}
    >>>     return indices
    >>> def get_spectral_activity(pre_calc_vars):
    >>>     ACTspfract_per_bin, ACTspcount_per_bin, ACTspmean_per_bin = (
                        maad.features.spectral_activity(pre_calc_vars['Sxx_dB_noNoise']))
    >>>     indices = {'ACTspfract_per_bin': ACTspfract_per_bin,
    >>>                'ACTspcount_per_bin':ACTspcount_per_bin,
    >>>                'ACTspmean_per_bin':ACTspmean_per_bin}
    >>>     return indices
    >>> acoustic_indices_methods = [get_aci, get_spectral_activity, get_spectral_events]
    >>>
    >>> df_temp = df.iloc[0:1]
    >>> segmented_df = utils.false_color_spectrogram_prepare_dataset(
    >>>     df_temp,
    >>>     datetime_col = 'timestamp_init',
    >>>     duration_col = 'duration',
    >>>     file_path_col = 'file_path',
    >>>     indices = ['acoustic_complexity_index', 'spectral_activity', 'spectral_events'],
    >>>     output_dir = './segmented_indices',
    >>>     store_audio_segments = True,
    >>>     unit = 'scale_02',
    >>>     acoustic_indices_methods = acoustic_indices_methods,
    >>>     pre_calculation_method = pre_calculation_method,
    >>>     temp_dir = os.path.abspath('./temp_ac_files/'),
    >>>     parallel = True
    >>> )
    >>>
    >>> fcs = visualizations.false_color_spectrogram_plot(
    >>>             segmented_df,
    >>>             datetime_col = 'start_time',
    >>>             indices = ['aci_per_bin', 'ACTspfract_per_bin', 'EVNspCount_per_bin'],
    >>>             display = True,
    >>>             unit = 'scale_02'
    >>>         )



    """

    # 0. Initial configuration
    # 0.1. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0:
        raise IndexError("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, (
            f"'{index}' is not in {df.columns}. "
            "Verify if it is correctly spelled and if it have been calculated already."
        )

    # 0.2. Verify if the unity is accepted
    available_units = [
        "scale_02",
        "scale_04",
        "scale_06",
        "scale_2",
        "scale_4",
        "scale_6",
        "scale_12",
        "scale_24",
        "scale_60",
    ]
    if unit not in available_units:
        raise Exception(
            f"""The unity {unit} is not available. """
            f"""The list of available unities is: {available_units}"""
        )

    # 1. Order original dataset by timestamp and create helper columns
    df = df.sort_values(by=datetime_col)
    trunc_unit = "min"
    if unit != "scale_60":
        trunc_unit = "s"
    df["timestamp"] = df[datetime_col].dt.floor(trunc_unit)

    # 2. Normalize index and create false color spectrogram

    fc_spectrogram = []
    for index in indices:
        ind = df[index].tolist()
        ind = np.asarray(
            ind
        ).T  # transpose the array to place frequencies are in y axis
        ind_normalized = (255 * (ind - ind.min()) / (ind.max() - ind.min())).astype(
            np.uint8
        )
        fc_spectrogram.append(ind_normalized)

    fc_spectrogram = np.asarray(fc_spectrogram)
    fc_spectrogram = np.transpose(fc_spectrogram, (1, 2, 0))

    # 3. Display false color spectrogram
    if display:
        _display_false_color_spectrogram(
            df,
            fc_spectrogram,
            indices,
            fig_size=kwargs["fig_size"] if "fig_size" in kwargs.keys() else None,
            tick_interval=(
                kwargs["tick_interval"] if "tick_interval" in kwargs.keys() else None
            ),
        )

    return fc_spectrogram
