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

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import maad


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
        assert (
            index in df.columns
        ), f"'{index}' is not in {df.columns}. "\
            "Verify if it is correctly spelled and if it have been calculated already."

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
                "There are more categories than available color, "\
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

    fig.update_layout(
        title="Radar Plot - Comparisson between indices", title_x=0.5
    )

    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height = fig_size["height"], width = fig_size["width"])
    fig.update_layout(polar = {"radialaxis": {"showticklabels": False}})

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
        assert group_by in df.columns, f"'{group_by}' is not in {df.column}"
        if len(indices) > 1:
            raise Exception(
                "Sorry, to group by some category, only one index is supported."
            )

    # 0.2. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0:
        raise Exception("Sorry, the indices list must be non empty.")
    for index in indices:
        assert (
            index in df.columns
        ), f"'{index}' is not in {df.columns}. "\
            "Verify if it is correctly spelled and if it have been calculated already."

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
        assert (
            index in df.columns
        ), f"'{index}' is not in {df.columns}. "\
            "Verify if it is correctly spelled and if it have been calculated already."

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
                "There are more categories than available color, "\
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
    s, fs = maad.sound.load(file_path)

    # 2. Calculate spectrogram
    sxx, tn, fn, _ = maad.sound.spectrogram(
        s,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
        verbose=verbose,
        mode=mode,
        window=window,
    )

    if mode == "psd":
        sxx_disp = maad.util.power2dB(sxx)
    if mode == "amplitude":
        sxx_disp = maad.util.amplitude2dB(sxx)
    if mode == "complex":
        sxx_disp = maad.util.amplitude2dB(sxx)

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
