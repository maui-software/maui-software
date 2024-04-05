"""
This module provides utilities for Exploratory Data Analysis (EDA) with an
emphasis on visual and summary outputs for categorical data within a DataFrame.
Utilizing Plotly for dynamic visualizations and FPDF for PDF report generation,
it supports the creation of summary cards and plots, enhancing data
understanding and presentation.

Features include:
- Generation of summary cards for data overview.
- Creation of various plots (scatter, bar, etc.) for data comparison and trend
  analysis.
- PDF report generation for easy sharing and presentation.

Exceptions:
- CategoryLimitError: Raised when an attempt is made to process more than the
  allowed number of categories.

Dependencies:
- plotly: For creating interactive plots.
- fpdf: For generating PDF reports.
- pandas: Assumed for DataFrame manipulation and input.

Functions:
- card_summary(df, categories, show_plot=True): Generates a summary card and
  plots for specified categories.

Note:
- This module is designed to work with pandas DataFrames and expects a specific
  structure/format for input data.
"""

import tempfile
import time

import pkg_resources
from fpdf import FPDF
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CategoryLimitError(Exception):
    """Exception raised when more than two categories are selected."""


def card_summary(df, categories, show_plot: bool = True):
    """
    Generates a summary card and plots for specified categories from a
    DataFrame.
    This function processes the input DataFrame to compute various statistics, including the
    number of samples, distinct days, total and mean duration (in minutes) of some activities.
    It also dynamically incorporates additional specified categories into its computations and
    visualizations. If enabled, a plot is generated using Plotly to visually represent these
    statistics alongside the categories specified.

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame containing at least the following columns: 'file_path', 'dt',
            and 'duration'. Additional columns should match the specified categories if any.
    categories : list of str
            A list of category names (column names in `df`) to include in the summary and plot.
            At most two categories can be specified.
    show_plot : bool, optional
            If True (default), the function will generate and show a Plotly plot representing the
            calculated statistics and specified categories. If False, no plot will be displayed.

    Raises
    ------
    Exception
            If more than two categories are specified, an exception is
            raised due to plotting limitations.

    Returns
    -------
    tuple
            Returns a tuple containing:
            
            - card_dict (dict): A dictionary with keys for 'n_samples',
                'distinct_days', 'total_time_duration',
                'mean_time_duration', and one key per category specified.
                The values are the respective
                computed statistics.

            - fig (plotly.graph_objs._figure.Figure): A Plotly figure object with indicators 
                for each of the statistics and categories specified.
                Only returned if `show_plot` is True.

    Notes
    -----
    The function is designed to work with data pertaining to
    durations and occurrences across
    different categories. It's particularly useful for analyzing time series or event data.
    The 'duration' column is expected to be in seconds.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> categories = ['landscape', 'environment']
    >>> card_dict, fig = eda.card_summary(df, categories)
    """

    if len(categories) > 2:
        raise CategoryLimitError("At most two categories should be selected.")

    df_count = df.nunique(axis=0)
    duration_mean = df["duration"].mean() / 60
    duration_total = df["duration"].sum() / 60

    card_dict = {
        "n_samples": df_count["file_path"],
        "distinct_days": df_count["dt"],
        "total_time_duration": duration_total,
        "mean_time_duration": duration_mean,
    }

    subplot_titles = ["Distinct Days", "Total Duration", "Mean Duration", "Samples"]

    for category in categories:
        card_dict[category] = df_count[category]
        subplot_titles.append(category)

    specs = [
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
    ]

    fig = make_subplots(rows=2, cols=3, subplot_titles=subplot_titles, specs=specs)

    trace0 = go.Indicator(
        mode="number",
        value=card_dict["distinct_days"],
        number={"suffix": ""},
        delta={"position": "top", "reference": 320},
        domain={"x": [0, 1], "y": [0, 1]},
    )
    fig.add_trace(trace0, 1, 1)

    trace1 = go.Indicator(
        mode="number",
        value=card_dict["total_time_duration"],
        number={"suffix": " min"},
        delta={"position": "top", "reference": 320},
        domain={"x": [0, 1], "y": [0, 1]},
    )
    fig.append_trace(trace1, 1, 2)

    trace2 = go.Indicator(
        mode="number",
        value=card_dict["mean_time_duration"],
        number={"suffix": " min"},
        delta={"position": "top", "reference": 320},
        domain={"x": [0, 1], "y": [0, 1]},
    )
    fig.append_trace(trace2, 1, 3)

    trace3 = go.Indicator(
        mode="number",
        value=card_dict["n_samples"],
        number={"prefix": ""},
        delta={"position": "top", "reference": 320},
        domain={"x": [0, 1], "y": [0, 1]},
    )
    fig.append_trace(trace3, 2, 1)

    i = 2
    j = 2

    for category in categories:

        trace_tmp = go.Indicator(
            mode="number",
            value=card_dict[category],
            number={"prefix": ""},
            delta={"position": "top", "reference": 320},
            domain={"x": [0, 1], "y": [0, 1]},
        )
        fig.append_trace(trace_tmp, i, j)

        j = (j % 3) + 1
        if j == 1:
            i = i + 1

    # fig.update_layout(paper_bgcolor = "lightgray")
    if show_plot:
        fig.show()

    return card_dict, fig


# ----------------------------------------------------------------------------


def heatmap_analysis(
    df,
    x_axis: str,
    y_axis: str,
    color_continuous_scale="Viridis",
    show_plot: bool = True,
):
    """
    Generates a heatmap to analyze the relationship between two categorical
    variables in a DataFrame.

    This function groups the data by the specified `x_axis` and `y_axis`
    categories, counts the occurrences of each group, and then creates a
    heatmap visualization of these counts using Plotly Express. The heatmap
    intensity is determined by the count of occurrences, with an option to
    customize the color scale.

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame containing the data to be analyzed.
            Must include the columns specified by `x_axis` and `y_axis`,
            as well as a 'file_path' column used for counting occurrences.
    x_axis : str
            The name of the column in `df` to be used as the x-axis in the heatmap.
    y_axis : str
            The name of the column in `df` to be used as the y-axis in the heatmap.
    color_continuous_scale : str, optional
            The name of the color scale to use for the heatmap. 
            Defaults to 'Viridis'. For more options, refer
            to Plotly's documentation on color scales.
    show_plot : bool, optional
            If True (default), displays the heatmap plot. 
            If False, the plot is not displayed but is still returned.

    Returns
    -------
    tuple
            A tuple containing:
            - df_group (pandas.DataFrame): A DataFrame with the grouped counts
            for each combination of `x_axis` and `y_axis` values.
            - fig (plotly.graph_objs._figure.Figure): A Plotly figure object containing the heatmap.

    Notes
    -----
    The 'file_path' column in the input DataFrame is used to count occurrences
    of each group formed by the specified `x_axis` and `y_axis` values.
    This function is useful for visualizing the distribution and
    relationship between two categorical variables.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> df_group, fig = eda.heatmap_analysis(df, 'landscape', 'environment')
    """

    df_group = df.groupby([x_axis, y_axis], as_index=False)["file_path"].count()
    df_group = df_group.rename(columns={"file_path": "count"})

    df_group_temp = df_group.pivot(index=x_axis, columns=y_axis, values="count")

    fig = px.imshow(
        df_group_temp,
        color_continuous_scale=color_continuous_scale,
        text_auto=True,
        title=f"""{x_axis} vs {y_axis} Heatmap""",
    )
    fig.update_layout(title_x=0.5)

    if show_plot:
        fig.show()

    return df_group, fig


# ----------------------------------------------------------------------------


def histogram_analysis(df, x_axis: str, category_column: str, show_plot: bool = True):
    """
    Generates a histogram plot for data distribution across a specified axis,
    optionally segmented by categories.

    This function creates a histogram to visualize the distribution of data
    in `df` along the `x_axis`, with data optionally segmented by
    `category_column`. The histogram's appearance, such as opacity and
    bar gap, is customizable. The plot is generated using Plotly Express
    and can be displayed in the notebook or IDE if `show_plot` is set to True.

    Parameters
    ----------
    df : pandas.DataFrame
            The DataFrame containing the data to plot.
            Must include the columns specified by `x_axis`
            and `category_column`.
    x_axis : str
            The name of the column in `df` to be used for the x-axis of the histogram.
    category_column : str
            The name of the column in `df` that contains categorical data for
            segmenting the histogram. Each category will be represented with
            a different color.
    show_plot : bool, optional
            If True (default), the generated plot will be immediately displayed.
            If False, the plot will not be displayed but will still be returned
            by the function.

    Returns
    -------
    plotly.graph_objs._figure.Figure
            The Plotly figure object for the generated histogram.
            This object can be further customized or saved after the function returns.

    Notes
    -----
    This function is designed to offer a quick and convenient way to visualize
    the distribution of data in a DataFrame along a specified axis.
    It is particularly useful for exploratory data analysis and
    for identifying patterns or outliers in dataset segments.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> fig = eda.histogram_analysis(df, 'landscape', 'environment')
    """

    fig = px.histogram(
        df,
        x=x_axis,
        color=category_column,
        opacity=0.7,
        title=f"""Amount of samples by {x_axis} and segmented by {category_column}""",
    )
    fig.update_layout(bargap=0.1, title_x=0.5)

    if show_plot:
        fig.show()

    return fig


# ----------------------------------------------------------------------------


def duration_analysis(df, category_column: str, duration_column: str, show_plot=True):
    """
    Generates a box plot visualizing the distribution of durations across different categories.

    This function takes a DataFrame and creates a box plot to analyze
    the distribution of durations (or any numerical data) across specified
    categories. The box plot provides a visual representation of the central
    tendency, dispersion, and skewness of the data and identifies outliers.

    Parameters
    ----------
    df : pandas.DataFrame
            The DataFrame containing the data to be analyzed.
            It should include at least two columns:
            one for the category and one for the duration
            (or any numerical data to be analyzed).
    category_column : str
            The name of the column in `df` that contains the categorical data. This column will be
            used to group the numerical data into different categories for the box plot.
    duration_column : str
            The name of the column in `df` that contains the numerical data to
            be analyzed. This data will be distributed into boxes according to
            the categories specified by `category_column`.
    show_plot : bool, optional
            If True (default), the function will display the generated box plot. If False, the plot
            will not be displayed, but the figure object will still be returned.

    Returns
    -------
    plotly.graph_objs._figure.Figure
            The generated Plotly figure object containing the box plot.
            This object can be used for further customization or to display
            the plot at a later time if `show_plot` is False.

    Notes
    -----
    The box plot generated by this function can help identify the range, interquartile range,
    median, and potential outliers within each category. This visual analysis is crucial for
    understanding the distribution characteristics of numerical data across different groups.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> fig = eda.duration_analysis(df, 'landscape', 'duration')
    """

    fig = px.box(
        df,
        x=category_column,
        y=duration_column,
        title=f"""Duration distribution by {category_column}""",
    )
    fig.update_layout(title_x=0.5)

    if show_plot:
        fig.show()

    return fig


# ----------------------------------------------------------------------------


def daily_distribution_analysis(
    df, date_column: str, category_column: str, show_plot=True
):
    """
    Analyzes and visualizes the daily distribution of samples by categories.

    This function generates a histogram that shows the distribution of samples over days, separated
    by a specified category. It provides insights into how the frequency of samples varies daily
    and according to the categories within the specified category column.

    Parameters
    ----------
    df : pandas.DataFrame
            The DataFrame containing the data to be analyzed. 
            It must include the specified `date_column` and `category_column`.
    date_column : str
            The name of the column in `df` that contains date information. The values in this column
            should be in a date or datetime format.
    category_column : str
            The name of the column in `df` that contains categorical data,
            which will be used to color the bars in the histogram.
    show_plot : bool, optional
            If True (default), the function will display the generated plot. If False, the plot will
            not be displayed but will still be returned.

    Returns
    -------
    plotly.graph_objs._figure.Figure
            A Plotly figure object representing the histogram of daily sample distribution by
            the specified category. The histogram bars are colored based on the categories
            in the `category_column`.

    Notes
    -----
    The function leverages Plotly for plotting, thus ensuring interactive plots that can be further
    explored in a web browser. It's particularly useful for time series data where understanding the
    distribution of events or samples over time and across different categories is crucial.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> fig = eda.daily_distribution_analysis(df, 'dt', 'landscape')
    """

    fig = px.histogram(
        df,
        x=date_column,
        color=category_column,
        opacity=0.7,
        title=f"""Ammount of samples by Day and {category_column}""",
    )
    fig.update_layout(bargap=0.1, title_x=0.5)

    if show_plot:
        fig.show()

    return fig


# ----------------------------------------------------------------------------


def duration_distribution(df, show_plot=True):
    """
    Generates a distribution plot for the 'duration' column in the provided DataFrame.

    This function creates a distribution plot, including a
    histogram and a kernel density estimate (KDE),
    for the 'duration' column in the input DataFrame. 
    It is designed to give a visual understanding of the
    distribution of duration values across the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
            The DataFrame containing the data to be analyzed. 
            It must include a column named 'duration',
            which contains numeric data.
    show_plot : bool, optional
            If True (default), the function will display the generated plot. If False, the plot will
            not be displayed but will still be returned.

    Returns
    -------
    plotly.graph_objs._figure.Figure
            A Plotly figure object representing the distribution plot of
            the 'duration' column. The plot includes both a histogram of
            the data and a kernel density estimate (KDE) curve.

    Notes
    -----
    The function uses Plotly's `create_distplot` function from the `plotly.figure_factory` module,
    offering a detailed visual representation of data distribution. It's particularly useful for
    analyzing the spread and skewness of numeric data. The KDE curve provides insight into the
    probability density of the durations, complementing the histogram's discrete bins.

    Examples
    --------
    >>> from maui import samples, eda
    >>> df = samples.get_leec_audio_sample()
    >>> fig = eda.duration_distribution(df)
    """

    group_labels = ["duration"]  # name of the dataset

    fig = ff.create_distplot([df["duration"].values], group_labels)
    fig.update_layout(bargap=0.005, title_text="Duration distribution", title_x=0.5)

    if show_plot:
        fig.show()

    return fig


# ----------------------------------------------------------------------------


class PDF(FPDF):
    """
    Internal class to organize the PDF generation
    """

    def footer(self):
        """
        Generate footer of the PDF.
        """
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(
            0,
            10,
            "Generated with <3 by Maui Software - Page " + str(self.page_no()),
            0,
            0,
            "C",
        )


def create_letterhead(pdf, width, image):
    """
    Generate title of the PDF.
    """
    pdf.image(image, 0, 0, width)


def create_title(pdf, title, subtitle=None):
    """
    Generate title of the PDF.
    """

    # Add main title
    pdf.set_font("Helvetica", "b", 20)
    pdf.ln(100)
    pdf.write(5, title)
    pdf.ln(15)

    if subtitle is not None:
        # Add subtitle
        pdf.set_font("Helvetica", "b", 16)
        pdf.write(5, subtitle)
        pdf.ln(10)

    # Add date of report
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(r=128, g=128, b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f"{today}")

    # Add line break
    pdf.ln(30)


def write_to_pdf(pdf, words):
    """
    Write data.
    """

    # Set text colour, font size, and font type
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.set_font("Helvetica", "", 12)

    pdf.write(5, words)


def write_subtitle(pdf, words):
    """
    Write subtitle.
    """

    # Set text colour, font size, and font type
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.set_font("Helvetica", "b", 14)

    pdf.write(5, words)


def export_file_names_summary_pdf_leec(
    df, file_name: str, analysis_title=None, width=210):
    """
    Export a summary PDF report with analysis of file names for LEEC project.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be analyzed.
    file_name : str
        Name of the output PDF file.
    analysis_title : str, optional
        Title of the analysis section in the PDF.
    width : int, optional
        Width of the PDF document in millimeters.

    Returns
    -------
    None

    Notes
    -----
    This function exports a summary PDF report with various analyses of file
    names for the LEEC project. 
    It includes landscape analysis, environment analysis, and duration
    analysis. 
    The PDF is created using the provided DataFrame `df` and saved with
    the specified `file_name`.

    Examples
    --------
    >>> export_file_names_summary_pdf_leec(df, 'summary_report.pdf', analysis_title='Audio Files Analysis')
    """

    categories = ['landscape', 'environment']

    with tempfile.TemporaryDirectory() as temp_dir:

        _, fig = card_summary(df, categories, show_plot=False)
        fig.write_image(f"""{temp_dir}/summary1.png""", height=300, width=1200)
        _, fig = heatmap_analysis(
            df,
            "landscape",
            "environment",
            color_continuous_scale="Viridis",
            show_plot=False,
        )
        fig.write_image(f"""{temp_dir}/summary2.png""")

        fig = histogram_analysis(df, "landscape", "environment", show_plot=False)
        fig.write_image(f"""{temp_dir}/landscape1.png""", height=400, width=1200)
        fig = duration_analysis(df, "landscape", "duration", show_plot=False)
        fig.write_image(f"""{temp_dir}/landscape2.png""", height=400, width=1200)
        fig = daily_distribution_analysis(df, "dt", "landscape", show_plot=False)
        fig.write_image(f"""{temp_dir}/landscape3.png""", height=400, width=1200)

        fig = histogram_analysis(df, "environment", "landscape", show_plot=False)
        fig.write_image(f"""{temp_dir}/environment1.png""", height=400, width=1200)
        fig = duration_analysis(df, "environment", "duration", show_plot=False)
        fig.write_image(f"""{temp_dir}/environment2.png""", height=400, width=1200)
        fig = daily_distribution_analysis(df, "dt", "environment", show_plot=False)
        fig.write_image(f"""{temp_dir}/environment3.png""", height=400, width=1200)

        fig = duration_distribution(df, show_plot=False)
        fig.write_image(f"""{temp_dir}/duration1.png""", height=400, width=1200)

        # Global Variables
        title = "Audio Files Exploratory Data Analysis"
        subtitle = analysis_title

        # Create PDF
        pdf = PDF()  # A4 (210 by 297 mm)

        # First Page of PDF

        # Add Page
        pdf.add_page()

        letterhead_cover = pkg_resources.resource_filename(
            "maui", "data/letterhead_cover.png"
        )
        letterhead = pkg_resources.resource_filename("maui", "data/letterhead.png")

        # Add lettterhead and title
        create_letterhead(pdf, width, letterhead_cover)
        create_title(pdf, title, subtitle)

        # Add table
        w = 200
        pdf.image(f"""{temp_dir}/summary1.png""", w=w, x=(width - w) / 2)
        pdf.ln(5)

        intro_text = """
        This report contains a brief exploratory data analysis """\
        "comprehending the data obtained by audio file names. "\
        "The objective is to present an overview of the acoustic """\
        "landscapes and environments of the recordings, as well as their duration. "\
        "Further analysis such as false color spectrograms and acoustic indices "\
        "summarization can be performed with Maui Sotware analysis and "\
        "visualization tools."
        write_to_pdf(pdf, intro_text)

        pdf.add_page()

        create_letterhead(pdf, width, letterhead)

        pdf.ln(20)
        write_subtitle(pdf, "1. Landscape Analysis")
        pdf.ln(20)
        pdf.image(f"""{temp_dir}/landscape1.png""", w=w, x=(width - w) / 2)
        pdf.ln(5)
        pdf.image(f"""{temp_dir}/landscape2.png""", w=w, x=(width - w) / 2)
        pdf.ln(5)
        pdf.image(f"""{temp_dir}/landscape3.png""", w=w, x=(width - w) / 2)
        pdf.ln(10)

        pdf.add_page()
        create_letterhead(pdf, width, letterhead)

        pdf.ln(20)
        write_subtitle(pdf, "2. Environment Analysis")
        pdf.ln(20)
        pdf.image(f"""{temp_dir}/environment1.png""", w=w, x=(width - w) / 2)
        pdf.ln(5)
        pdf.image(f"""{temp_dir}/environment2.png""", w=w, x=(width - w) / 2)
        pdf.ln(5)
        pdf.image(f"""{temp_dir}/environment3.png""", w=w, x=(width - w) / 2)
        pdf.ln(10)

        pdf.add_page()
        create_letterhead(pdf, width, letterhead)

        pdf.ln(20)
        write_subtitle(pdf, "3. Duration Analysis")
        pdf.ln(20)
        pdf.image(f"""{temp_dir}/duration1.png""", w=w, x=(width - w) / 2)

        pdf.output(file_name, "F")
