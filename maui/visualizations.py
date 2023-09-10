import math
import numpy as np
import pandas as pd
import copy
import warnings
import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import maad


def indices_radar_plot(df, indices:list, agg_type:str, group_by:list=None, max_cols:int=3, fig_size:dict=None, show_plot:bool=True):
    """
    Create a radar plot (spider plot) comparing multiple indices from a DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data.

    indices : list of str
        A list of column names representing the indices to be compared.

    agg_type : str
        The type of aggregation to be applied to the data. Supported options are:
        - 'mean': Calculate the mean value for each index.
        - 'median': Calculate the median value for each index.
        - 'stddev': Calculate the standard deviation for each index.
        - 'var': Calculate the variance for each index.
        - 'max': Calculate the maximum value for each index.
        - 'min': Calculate the minimum value for each index.

    group_by : list of str, optional
        A list of column names by which the data should be grouped. Each unique value
        in these columns will be represented as a separate category in the radar plot.
        Defaults to None, indicating no grouping.

    max_cols : int, optional
        The maximum number of columns for the subplots in the radar plot grid. This
        parameter is only relevant when grouping data. Defaults to 3.

    fig_size : dict, optional
        A dictionary specifying the height and width of the radar plot figure. It
        should have 'height' and 'width' keys. Defaults to None.

    show_plot : bool, optional
        Whether to display the radar plot immediately using Plotly. Defaults to True.

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The Plotly figure object representing the radar plot.

    Raises:
    -------
    AssertionError
        - If 'agg_type' is not one of the supported aggregation options.
        - If a column specified in 'group_by' is not found in the DataFrame.
        - If 'indices' is empty or contains indices not present in the DataFrame.
        - If 'fig_size' dictionary is missing 'height' or 'width' keys.

    Exception
        - If the maximum number of categories to group by exceeds 2.

    Warning
        - If there are more unique categories in 'group_by' than available colors.

    Example:
    --------
    # Create a radar plot comparing 'index1' and 'index2' aggregated by 'category'
    fig = indices_radar_plot(data_df, ['index1', 'index2'], 'mean', group_by=['category'])
    """
    
    # 0. Initial configuration
    # 0.1. Verify if agg_type is available
    agg_options = ['mean', 'median', 'stddev', 'var', 'max', 'min']
    assert agg_type in agg_options, f"'{agg_type}' is not in {agg_options}"
    
    # 0.2. Verify if group_by column is available
    if group_by is not None:
        for col in group_by:
            assert col in df.columns, f"'{col}' is not in {df.column}"
            
        # 0.2.1. Verify if there is a maximum of two categories to group by
        if len(group_by) > 2:
            raise Exception("Sorry, the maximum categories to group by is 2")
        
    # 0.3. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0: raise Exception("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, f"'{index}' is not in {df.columns}. Verify if it is correctly spelled and if it have been calculated already."
        
    # 0.4. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if 'height' not in fig_size.keys() or 'width' not in fig_size.keys(): raise Exception("fig_size must contain width and height keys.")
    
    # 0.5 Create a copy of the dataframe
    df_unpivot = copy.deepcopy(df)
    
    #--------------------------------------------

    # 1. Normalize columns to better suit the plot
    for index in indices:
        df_unpivot[index] = (df_unpivot[index]-df_unpivot[index].min())/(df_unpivot[index].max()-df_unpivot[index].min())
        
    #--------------------------------------------
        
    # 2. Unpivot dataframe
    df_unpivot = pd.melt(df_unpivot, id_vars=group_by, value_vars=indices)
    
    #--------------------------------------------
        
    # 3. Aggregate data
    gb_cols = ['variable']
    if group_by is not None:
        for col in group_by: gb_cols.append(col)
    
    if agg_type == 'mean':
        df_unpivot = df_unpivot.groupby(gb_cols).mean().reset_index()
    if agg_type == 'median':
        df_unpivot = df_unpivot.groupby(gb_cols).median().reset_index()
    if agg_type == 'stddev':
        df_unpivot = df_unpivot.groupby(gb_cols).std().reset_index()
    if agg_type == 'var':
        df_unpivot = df_unpivot.groupby(gb_cols).var().reset_index()
    if agg_type == 'max':
        df_unpivot = df_unpivot.groupby(gb_cols).var().reset_index()
    if agg_type == 'min':
        df_unpivot = df_unpivot.groupby(gb_cols).var().reset_index()
    
    df_unpivot = df_unpivot.rename(columns={'variable': 'index'})
    
    #--------------------------------------------
        
    # 4. Plot data
    
    n_cols = 1
    n_rows = 1

    if group_by is not None and len(group_by) > 1:
        n_cols = min(len(list(df[group_by[0]].unique())), max_cols)
        n_rows = math.ceil(len(list(df[group_by[0]].unique())) / max_cols) 
        
        

    fig = make_subplots(
            rows=n_rows, cols=n_cols, 
            specs=[[{'type': 'polar'}]*n_cols]*n_rows,
            subplot_titles=[' ']*n_cols*n_rows
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
            
            fig.add_trace( go.Scatterpolar(r=r, theta=theta, mode='lines'), 
                          row=1, col=1)
    else:
        if len(group_by) == 1: lables_list = list(df[group_by[0]].unique())
        else: 
            lables_list = list(df[group_by[1]].unique())
        if len(lables_list) > len(colors):  warnings.warn("There are more categories than available color, some categories may use the same color")
            
        for ind, category in enumerate(list(df[group_by[0]].unique())): 


            df_tmp = copy.copy(df_unpivot[df_unpivot[group_by[0]] == category])

            if len(group_by) == 1:
                r = list(df_tmp["value"])
                r.append(r[0])
                theta = list(df_tmp["index"])
                theta.append(theta[0])
                
                fig.add_trace( go.Scatterpolar(name = category, r=r, theta=theta, mode='lines', legendgroup=category, showlegend=showlegend), 
                              row=row, col=col)
                if category in lables_list: lables_list.remove(category)
                if len(lables_list) == 0: showlegend = False
                    
            else:

                for j, filter_col in enumerate(list(df[group_by[1]].unique())):

                    df_tmp_final = copy.copy(df_tmp[df_tmp[group_by[1]] == filter_col])
                    
        
                    r = list(df_tmp_final["value"])
                    if len(r) > 0:
                        r.append(r[0])
                        theta = list(df_tmp_final["index"])
                        theta.append(theta[0])

                        if filter_col not in lables_list: showlegend = False
                        else: 
                            lables_list.remove(filter_col)
                            showlegend = True
                            
                        
                        fig.add_trace( go.Scatterpolar(name = filter_col, r=r, theta=theta, mode='lines', legendgroup=filter_col, 
                                                       showlegend=showlegend, line_color = colors[j % len(colors)]), 
                                      row=row, col=col)   
                        fig.update_polars(radialaxis_showticklabels=False)
                            
                fig.layout.annotations[ind]['text'] = category
                fig.layout.annotations[ind]['yshift'] = 25
                
                if col >= max_cols:
                    col = 1
                    row += 1
                else:
                    col += 1
                    
    fig.update_layout(
        title=f'''Radar Plot - Comparisson between indices''',
        title_x=0.5
    )
    
    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size['height'], width=fig_size['width'])
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    
    if show_plot:
        fig.show()
    return fig
    
# --------------------------------------------------------------------------------------------------------------------------------

def indices_histogram_plot(df, indices:list, group_by:str=None, max_cols:int=3, fig_size:dict=None, show_plot:bool=True):
    """
    Create a histogram plot comparing the distribution of selected indices in a DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data.

    indices : list of str
        A list of column names representing the indices to be compared.

    group_by : str, optional
        A column name by which the data should be grouped. Each unique value in this
        column will be represented as a separate category in the histogram plot.
        Defaults to None, indicating no grouping.

    max_cols : int, optional
        The maximum number of columns for the subplots in the histogram plot grid.
        Defaults to 3.

    fig_size : dict, optional
        A dictionary specifying the height and width of the histogram plot figure. It
        should have 'height' and 'width' keys. Defaults to None.

    show_plot : bool, optional
        Whether to display the histogram plot immediately using Plotly. Defaults to True.

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The Plotly figure object representing the histogram plot.

    Raises:
    -------
    AssertionError
        - If 'group_by' is specified, but the column is not found in the DataFrame.
        - If 'indices' is empty or contains indices not present in the DataFrame.
        - If 'fig_size' dictionary is missing 'height' or 'width' keys.

    Exception
        - If 'group_by' is specified, but more than one index is provided for comparison.

    Example:
    --------
    # Create a histogram plot for the distribution of 'index1'
    fig = indices_histogram_plot(data_df, ['index1'])

    # Create a grouped histogram plot for the distribution of 'index2' by 'category'
    fig = indices_histogram_plot(data_df, ['index2'], group_by='category')
    """

    
    # 0. Initial configuration
    # 0.1. Verify if group_by column is available
    if group_by is not None:
        assert group_by in df.columns, f"'{group_by}' is not in {df.column}"
        if len(indices) > 1: raise Exception("Sorry, to group by some category, only one index is supported.")
        
            
        
    # 0.2. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0: raise Exception("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, f"'{index}' is not in {df.columns}. Verify if it is correctly spelled and if it have been calculated already."
        
    # 0.3. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if 'height' not in fig_size.keys() or 'width' not in fig_size.keys(): raise Exception("fig_size must contain width and height keys.")
    
        
    #--------------------------------------------
    
    # 1. Plot data

    if group_by is None:
        n_cols = min(len(indices), max_cols)
        n_rows = math.ceil(len(indices) / max_cols) 
    else:
        n_cols = min(len(df[group_by].unique()), max_cols)
        n_rows = math.ceil(len(df[group_by].unique()) / max_cols) 

    fig = make_subplots(
            rows=n_rows, cols=n_cols, 
            specs=[[{'type': 'histogram'}]*n_cols]*n_rows,
            subplot_titles=[' ']*n_cols*n_rows
        )
    
    col = 1
    row = 1
    showlegend = True
    
    colors = px.colors.qualitative.Plotly
    if group_by is not None: lables_list = list(df[group_by].unique())
    
    # add one trace for each index
    if group_by is None: 
        for i, index in enumerate(indices):

            fig.add_trace( go.Histogram(name = index, x=df[index], showlegend=showlegend), 
                                         row=row, col=col)   
            fig.layout.annotations[i]['text'] = index
            fig.layout.annotations[i]['yshift'] = 25

            if col >= max_cols:
                col = 1
                row += 1
            else:
                col += 1
    else:
        for i, category in enumerate(df[group_by].unique()):
            df_index = df[df[group_by] == category]

            fig.add_trace( go.Histogram(name = index, x=df_index[indices[0]], showlegend=showlegend), 
                                         row=row, col=col)   
            fig.layout.annotations[i]['text'] = category
            fig.layout.annotations[i]['yshift'] = 25

            if col >= max_cols:
                col = 1
                row += 1
            else:
                col += 1
    
    fig.update_layout(
        title=f'''Histogram Plot - Distribution of selected indices''',
        title_x=0.5
    )
        
        
    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size['height'], width=fig_size['width'])
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    
    if show_plot:
        fig.show()
    return fig
        
# --------------------------------------------------------------------------------------------------------------------------------

def indices_violin_plot(df, indices:list, group_by:str=None, fig_size:dict=None, show_plot:bool=True):
    """
    Create a violin plot comparing the distribution of selected indices in a DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data.

    indices : list of str
        A list of column names representing the indices to be compared.

    group_by : str, optional
        A column name by which the data should be grouped. Each unique value in this
        column will be represented as a separate category in the violin plot.
        Defaults to None, indicating no grouping.

    fig_size : dict, optional
        A dictionary specifying the height and width of the violin plot figure. It
        should have 'height' and 'width' keys. Defaults to None.

    show_plot : bool, optional
        Whether to display the violin plot immediately using Plotly. Defaults to True.

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The Plotly figure object representing the violin plot.

    Raises:
    -------
    AssertionError
        - If 'group_by' is specified, but the column is not found in the DataFrame.
        - If 'indices' is empty or contains indices not present in the DataFrame.
        - If 'fig_size' dictionary is missing 'height' or 'width' keys.

    Example:
    --------
    # Create a violin plot for the distribution of 'index1'
    fig = indices_violin_plot(data_df, ['index1'])

    # Create a grouped violin plot for the distribution of 'index2' by 'category'
    fig = indices_violin_plot(data_df, ['index2'], group_by='category')
    """
 
    
    # 0. Initial configuration
    # 0.1. Verify if group_by column is available
    if group_by is not None:
        assert group_by in df.columns, f"'{group_by}' is not in {df.column}"
        

    # 0.2. Verify if the select indices have been already calculated
    if indices is None or len(indices) == 0: raise Exception("Sorry, the indices list must be non empty.")
    for index in indices:
        assert index in df.columns, f"'{index}' is not in {df.columns}. Verify if it is correctly spelled and if it have been calculated already."
        
    # 0.4. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if 'height' not in fig_size.keys() or 'width' not in fig_size.keys(): raise Exception("fig_size must contain width and height keys.")
    
    
    #--------------------------------------------
    # 2. Plot data
    
    n_cols = 1
    n_rows = len(indices)

    fig = make_subplots(
            rows=n_rows, cols=n_cols, 
            specs=[[{'type': 'histogram'}]*n_cols]*n_rows,
            subplot_titles=[' ']*n_cols*n_rows
        )
    
    showlegend = True
    
    violin_width = 0
    if fig_size is not None and group_by is not None:
        violin_width = (fig_size['width'] / (len(list(df[group_by].unique())))) / fig_size['width']
    elif fig_size is not None and group_by is None:
        violin_width = 0.3
        
    colors = px.colors.qualitative.Plotly
    
    if group_by is not None: 
        lables_list = list(df[group_by].unique())
        categories = list(df[group_by].unique())
        
        if len(lables_list) > len(colors):  warnings.warn("There are more categories than available color, some categories may use the same color")
        
    
    # add one trace for each index
    for i, index in enumerate(indices):
        
        if group_by is not None:           
            
            
            for j, lab in enumerate(categories):
        
                fig.add_trace(go.Violin(x=df[group_by][df[group_by] == lab],
                                        y=df[index][df[group_by] == lab],
                                        name=lab,
                                        box_visible=True,
                                        meanline_visible=True,
                                        points='all',
                                        scalemode='width',
                                        width=violin_width,
                                        legendgroup=lab,
                                        showlegend=showlegend,
                                        marker_color=colors[j % len(colors)],
                                        line_color=colors[j % len(colors)]
                                       ), row=i+1, col=1)
            
                if lab in lables_list: lables_list.remove(lab)
                if len(lables_list) == 0: showlegend = False
        else:
            fig.add_trace(go.Violin(y=df[index],
                                    name='',
                                    box_visible=True,
                                    meanline_visible=True,
                                    points='all',
                                    scalemode='width',
                                    width=violin_width,
                                    showlegend=False
                                   ), row=i+1, col=1)
            
        fig.layout.annotations[i]['text'] = index
        fig.layout.annotations[i]['yshift'] = 25
        
    fig.update_layout(
        title=f'''Violin Plot - Distribution of selected indices''',
        title_x=0.5
    )

        
        
    fig.layout.autosize = True
    if fig_size is not None:
        fig.update_layout(height=fig_size['height'], width=fig_size['width'])
        
    if show_plot:
        fig.show()
    return fig
        
# --------------------------------------------------------------------------------------------------------------------------------
    
def spectrogram_plot(file_path:str, mode:str=None, window:str='hann', nperseg:int=1024, noverlap:int=None, verbose:bool=False, fig_size:dict=None, show_plot:bool=True):
    """
    Create and display a spectrogram plot for an audio file using the scikit-maad package.

    Parameters:
    -----------
    file_path : str
        The path to the audio file to be processed.

    mode : str, optional
        The mode for spectrogram computation. Supported options are:
        - 'psd': Power Spectral Density (default).
        - 'amplitude': Amplitude.
        - 'complex': Complex representation.
        Defaults to 'psd'.

    window : str, optional
        The window function to be applied to the signal. Defaults to 'hann'.

    nperseg : int, optional
        The number of data points used in each block for the FFT. Defaults to 1024.

    noverlap : int, optional
        The number of points of overlap between blocks. If None, 50% overlap is used by default.

    verbose : bool, optional
        Whether to display verbose output during spectrogram computation. Defaults to False.

    fig_size : dict, optional
        A dictionary specifying the height and width of the spectrogram plot figure.
        It should have 'height' and 'width' keys. Defaults to None.

    show_plot : bool, optional
        Whether to display the spectrogram plot immediately using Plotly. Defaults to True.

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The Plotly figure object representing the spectrogram plot.

    Raises:
    -------
    AssertionError
        - If 'mode' is not one of the supported mode options.
        - If 'fig_size' dictionary is missing 'height' or 'width' keys.

    Example:
    --------
    # Create a spectrogram plot for an audio file with the default settings
    fig = spectrogram_plot("audio_file.wav")

    # Create a spectrogram plot with custom parameters
    fig = spectrogram_plot("audio_file.wav", mode='amplitude', window='hamming', nperseg=512, noverlap=256)
    """
    
    # 0. Validations
    # 0.1. available modes 
    mode_options = ['psd', 'mean', 'complex']
    assert mode in mode_options, f"'{mode}' is not in {mode_options}"
    
    # 0.2. Verify if fig_size is correctly defined (has two keys, height and width)
    if fig_size is not None:
        if 'height' not in fig_size.keys() or 'width' not in fig_size.keys(): raise Exception("fig_size must contain width and height keys.")
    
    #--------------------------------------------
    
    # 1. Load file
    s, fs = maad.sound.load(file_path)
    
    # 2. Calculate spectrogram
    Sxx,tn,fn,ext = maad.sound.spectrogram(s, fs, nperseg=nperseg, noverlap=noverlap, verbose=verbose, mode=mode, window=window) 
    
    if mode == 'psd':
        Sxx_disp = maad.util.power2dB(Sxx)
    if mode == 'amplitude':
        Sxx_disp = maad.util.amplitude2dB(Sxx)
    if mode == 'complex':
        Sxx_disp = maad.util.amplitude2dB(Sxx)
    
    fig_size = {'height':500, 'width':1200}

    fig = go.Figure(data=go.Heatmap(
            z=Sxx_disp,
            x=tn,
            y=fn,
            colorscale='gray',
            hoverinfo=None
          ))
    
    fig.update_layout(
        title=f'''Spectrogram generate from the file {os.path.basename(file_path)}''',
        title_x=0.5
    )
    
    if fig_size is not None:
            fig.update_layout(height=fig_size['height'], width=fig_size['width'])
            
    if show_plot:
        fig.show()
    return fig
    
    
    
    