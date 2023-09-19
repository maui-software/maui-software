import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import fpdf
from fpdf import FPDF
import time

import shutil
import os
import pkg_resources  

def card_summary(df, show_plot:bool=True):
	"""
	Generate a summary card for a DataFrame.

	This function calculates and displays summary statistics for a DataFrame,
	including the number of samples, landscapes, environments, distinct days, total time duration,
	and mean time duration.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame for which the summary statistics will be calculated.
	show_plot : bool, optional
		Whether to display the summary card plot. Default is True.

	Returns
	-------
	card_dict : dict
		A dictionary containing the calculated summary statistics.
	fig : plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the summary card plot.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> df = io.get_audio_info(audio_dir, store_duration=1, perc_sample=0.01)
	>>> summary, fig = eda.card_summary(df)
	>>> print(summary)
	{
		'n_samples': 120,
		'n_landscapes': 6,
		'n_environments': 3,
		'distinct_days': 18,
		'total_time_duration': 120.0,
		'mean_time_duration': 1.5
	}

	Notes
	-----
	- The summary statistics include the number of unique values in columns 'file_path',
	  'landscape', 'environment', and 'dt', as well as the mean time duration in minutes.
	- The 'show_plot' parameter controls the display of the summary card plot.
	"""

	df_count = df.nunique(axis=0)
	duration_mean = df['duration'].mean() / 60
	duration_total = df['duration'].sum() / 60
	
	card_dict = {
		'n_samples': df_count['file_path'],
		'n_landscapes': df_count['landscape'],
		'n_environments': df_count['environment'],
		'distinct_days': df_count['dt'],
		'total_time_duration': duration_total,
		'mean_time_duration': duration_mean
	}
	

	fig = make_subplots(rows=2, cols=3, subplot_titles=("Samples", "Landscapes", "Environments", "Distinct Days", "Total Duration", "Mean Duration"), 
						specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}], [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]])
	# specs=[[{'type': 'indicator'}, {}], [{‘colspan’: 2}, None]]

	trace0 = go.Indicator(
		mode = "number",
		value = card_dict['n_samples'],
		number = {'prefix': ""},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	trace1 = go.Indicator(
		mode = "number",
		value = card_dict['n_landscapes'],
		number = {'prefix': ""},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	trace2 = go.Indicator(
		mode = "number",
		value = card_dict['n_environments'],
		number = {'prefix': ""},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	trace3 = go.Indicator(
		mode = "number",
		value = card_dict['distinct_days'],
		number = {'suffix': ""},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	trace4 = go.Indicator(
		mode = "number",
		value = card_dict['total_time_duration'],
		number = {'suffix': " min"},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	trace5 = go.Indicator(
		mode = "number",
		value = card_dict['mean_time_duration'],
		number = {'suffix': " min"},
		delta = {'position': "top", 'reference': 320},
		domain = {'x': [0, 1], 'y': [0, 1]})

	fig.add_trace(trace0, 1, 1)
	fig.append_trace(trace1, 1, 2)
	fig.append_trace(trace2, 1, 3)
	fig.append_trace(trace3, 2, 1)
	fig.append_trace(trace4, 2, 2)
	fig.append_trace(trace5, 2, 3)

	# fig.update_layout(paper_bgcolor = "lightgray")
	if show_plot:
		fig.show()
	
	return card_dict, fig

#-----------------------------------------------------------------------------------------------------------------------------------

def landscape_environment_heatmap(df, show_plot:bool = True):
	"""
	Generate a heatmap of landscapes vs. environments.

	This function calculates the count of samples for each combination of landscapes and environments
	and displays the information as a heatmap.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for the heatmap.
	show_plot : bool, optional
		Whether to display the heatmap plot. Default is True.

	Returns
	-------
	pd.DataFrame
		A DataFrame containing the count of samples for each landscape-environment combination.
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the landscape vs. environment heatmap.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> heatmap_data, fig = eda.landscape_environment_heatmap(df)


	Notes
	-----
	- The function groups the input DataFrame 'df' by 'landscape' and 'environment', counts the number of samples for each combination, and displays the result as a heatmap.
	- The 'show_plot' parameter controls the display of the heatmap plot.


	"""
	

	df_group = df.groupby(['landscape', 'environment'], as_index=False)['file_path'].count()
	df_group = df_group.rename(columns={"file_path": "count"})
	
	
	df_group_temp = df_group.pivot(index='landscape', columns='environment', values='count')

	fig = px.imshow(df_group_temp, color_continuous_scale='Viridis', text_auto=True, title='Landscape vs Environment Heatmap')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return df_group, fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_histogram(df, show_plot:bool=True):
	"""
	Generate a histogram of the amount of samples by landscape.

	This function creates a histogram showing the distribution of samples across different landscapes,
	with color differentiation based on the environment.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the histogram.
	show_plot : bool, optional
		Whether to display the generated histogram plot. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the histogram of samples by landscape.

	Examples
	--------
	>>> import pandas as pd
	>>> df = pd.read_csv('data.csv')
	>>> fig = eda.plot_landscape_histogram(df)

	Notes
	-----
	- The function uses Plotly Express to create a histogram that displays the distribution of samples
	  across different landscapes, with color coding based on the environment.
	- The 'show_plot' parameter controls whether the histogram plot is displayed or not.
	"""

	fig = px.histogram(df, x="landscape", color="environment", opacity=0.7, title='Ammount of samples by Landscape')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_duration(df, show_plot=True):
	"""
	Generate a box plot of duration distribution by landscape.

	This function creates a box plot showing the distribution of audio duration across different landscapes.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the box plot.
	show_plot : bool, optional
		Whether to display the generated box plot. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the box plot of duration distribution by landscape.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_landscape_duration(df)

	Notes
	-----
	- The function uses Plotly Express to create a box plot that displays the distribution of audio duration
	  across different landscapes.
	- The 'show_plot' parameter controls whether the box plot is displayed or not.
	"""


	fig = px.box(df, x="landscape", y="duration", title='Duration distribution by Landscape')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_daily_distribution(df, show_plot=True):
	"""
	Generate a histogram of sample distribution by day and landscape.

	This function creates a histogram showing the distribution of audio samples by day and landscape.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the histogram.
	show_plot : bool, optional
		Whether to display the generated histogram. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the histogram of sample distribution by day and landscape.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_landscape_daily_distribution(df)

	Notes
	-----
	- The function uses Plotly Express to create a histogram that displays the distribution of audio samples
	  by day and landscape.
	- The 'show_plot' parameter controls whether the histogram is displayed or not.
	"""


	fig = px.histogram(df, x="dt", color="landscape", opacity=0.7, title='Ammount of samples by Day and Landscape')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_histogram(df, show_plot=True):
	"""
	Generate a histogram of sample distribution by environment.

	This function creates a histogram showing the distribution of audio samples by environment, optionally
	differentiated by landscape.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the histogram.
	show_plot : bool, optional
		Whether to display the generated histogram. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the histogram of sample distribution by environment.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_environment_histogram(df)

	Notes
	-----
	- The function uses Plotly Express to create a histogram that displays the distribution of audio samples
	  by environment.
	- If the 'landscape' column is present in the DataFrame, the histogram will differentiate the samples
	  by landscape.
	- The 'show_plot' parameter controls whether the histogram is displayed or not.
	"""
	

	fig = px.histogram(df, x="environment", color="landscape", opacity=0.7, title='Ammount of samples by Environment')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()

	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_duration(df, show_plot=True):
	"""
	Generate a box plot of audio sample durations by environment.

	This function creates a box plot showing the distribution of audio sample durations by environment.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the box plot.
	show_plot : bool, optional
		Whether to display the generated box plot. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the box plot of audio sample durations by environment.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_environment_duration(df)

	Notes
	-----
	- The function uses Plotly Express to create a box plot that displays the distribution of audio sample
	  durations by environment.
	- The 'show_plot' parameter controls whether the box plot is displayed or not.
	"""

	fig = px.box(df, x="environment", y="duration", title='Duration distribution by Environment')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_daily_distribution(df, show_plot=True):
	"""
	Generate a histogram of audio sample counts by day and environment.

	This function creates a histogram showing the distribution of audio sample counts by day and environment.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the histogram.
	show_plot : bool, optional
		Whether to display the generated histogram. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the histogram of audio sample counts by day and environment.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_environment_daily_distribution(df)
	>>> fig.show()

	Notes
	-----
	- The function uses Plotly Express to create a histogram that displays the distribution of audio sample
	  counts by day and environment.
	- The 'show_plot' parameter controls whether the histogram is displayed or not.
	"""

	fig = px.histogram(df, x="dt", color="environment", opacity=0.7, title='Ammount of samples by Day and Environment')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_duration_distribution(df, show_plot=True):
	"""
	Generate a distribution plot of audio sample durations.

	This function creates a distribution plot (histogram and kernel density estimate) of audio sample durations.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data to be used for generating the distribution plot.
	show_plot : bool, optional
		Whether to display the generated distribution plot. Default is True.

	Returns
	-------
	plotly.graph_objs._figure.Figure
		A Plotly Figure object representing the distribution plot of audio sample durations.

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> fig = eda.plot_duration_distribution(df)
	>>> fig.show()

	Notes
	-----
	- The function uses Plotly Figure Factory to create a distribution plot that displays the distribution of
	  audio sample durations.
	- The 'show_plot' parameter controls whether the distribution plot is displayed or not.
	"""
	

	group_labels = ['duration'] # name of the dataset

	fig = ff.create_distplot([df['duration'].values], group_labels)
	fig.update_layout(bargap=0.005, title_text='Duration distribution', title_x=0.5)

	if show_plot:
		fig.show()

	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

class PDF(FPDF):

	def footer(self):
		self.set_y(-15)
		self.set_font('Helvetica', 'I', 8)
		self.set_text_color(128)
		self.cell(0, 10, "Generated with <3 by Maui Software - Page " + str(self.page_no()), 0, 0, 'C')



def create_letterhead(pdf, WIDTH, image):
	pdf.image(image, 0, 0, WIDTH)

def create_title(pdf, title, subtitle=None):
	
	# Add main title
	pdf.set_font('Helvetica', 'b', 20)  
	pdf.ln(100)
	pdf.write(5, title)
	pdf.ln(15)
	
	if subtitle is not None:
		# Add subtitle
		pdf.set_font('Helvetica', 'b', 16)  
		pdf.write(5, subtitle)
		pdf.ln(10)
	
	# Add date of report
	pdf.set_font('Helvetica', '', 14)
	pdf.set_text_color(r=128,g=128,b=128)
	today = time.strftime("%d/%m/%Y")
	pdf.write(4, f'{today}')
	
	# Add line break
	pdf.ln(30)

def write_to_pdf(pdf, words):
	
	# Set text colour, font size, and font type
	pdf.set_text_color(r=0,g=0,b=0)
	pdf.set_font('Helvetica', '', 12)
	
	pdf.write(5, words)

def write_subtitle(pdf, words):
	
	# Set text colour, font size, and font type
	pdf.set_text_color(r=0,g=0,b=0)
	pdf.set_font('Helvetica', 'b', 14)
	
	pdf.write(5, words)

def export_file_names_summary_pdf(df, file_name, analysis_title=None, width=210, hight=297):
	"""
	Export a summary of audio file names and analysis results to a PDF.

	This function generates a PDF report summarizing information and analysis results from audio file names, including
	sample counts, landscape, environment, duration, and distribution plots.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing audio file name data.
	file_name : str
		The name of the PDF file to be generated.
	analysis_title : str, optional
		A custom title for the analysis section of the report. Default is None.
	width : int, optional
		The width of the PDF page in millimeters. Default is 210 (A4 paper width).
	height : int, optional
		The height of the PDF page in millimeters. Default is 297 (A4 paper height).

	Returns
	-------
	None

	Examples
	--------
	>>> from maui import samples, eda
	>>> df = samples.get_leec_audio_sample()
	>>> eda.export_file_names_summary_pdf(df, 'audio_summary.pdf', 'Audio Data Analysis')

	Notes
	-----
	- The function generates a PDF report containing summary statistics, heatmaps, histograms, box plots, and distribution plots
	  related to audio file names and their attributes.
	- The 'analysis_title' parameter allows you to customize the title of the analysis section in the report.
	- The 'width' and 'height' parameters control the dimensions of the PDF page. The default dimensions are for A4 paper.
	"""
	
	
	if not os.path.exists("images_summary_pdf_temp"):
		os.mkdir("images_summary_pdf_temp")
	
	card_dict, fig = card_summary(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/summary1.png", height=300, width=1200)
	df_group, fig = landscape_environment_heatmap(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/summary2.png")

	fig = plot_landscape_histogram(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/landscape1.png", height=400, width=1200)
	fig = plot_landscape_duration(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/landscape2.png", height=400, width=1200)
	fig = plot_landscape_daily_distribution(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/landscape3.png", height=400, width=1200)

	fig = plot_environment_histogram(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/environment1.png", height=400, width=1200)
	fig = plot_environment_duration(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/environment2.png", height=400, width=1200)
	fig = plot_environment_daily_distribution(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/environment3.png", height=400, width=1200)

	fig = plot_duration_distribution(df, show_plot=False)
	fig.write_image("images_summary_pdf_temp/duration1.png", height=400, width=1200)

	# Global Variables
	TITLE = "Audio Files Exploratory Data Analysis"
	SUBTITLE = analysis_title
	WIDTH = width

	# Create PDF
	pdf = PDF() # A4 (210 by 297 mm)


	'''
	First Page of PDF
	'''
	# Add Page
	pdf.add_page()

	letterhead_cover = pkg_resources.resource_filename('maui', 'data/letterhead_cover.png')
	letterhead = pkg_resources.resource_filename('maui', 'data/letterhead.png')

	# with pkg_resources.resource_filename('tempfile', 'data/letterhead_cover.png') as f:
	# 	letterhead_cover = f

	# with pkg_resources.resource_filename('tempfile', 'data/letterhead.png') as f:
	# 	letterhead = f

	# Add lettterhead and title
	create_letterhead(pdf, WIDTH, letterhead_cover)
	create_title(pdf, TITLE, SUBTITLE)


	# Add table
	w = 200
	pdf.image("images_summary_pdf_temp/summary1.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(5)

	intro_text = """
	This report contains a brief exploratory data analysis comprehending the data obtained by audio file names. 
	The objective is to present an overview of the acoustic landscapes and environments of the recordings, as well as their duration. 
	Further analysis such as false color spectrograms and acoustic indices summarization can be performed with Maui Sotware analysis and visualization tools."""
	write_to_pdf(pdf, intro_text)


	pdf.add_page()

	create_letterhead(pdf, WIDTH, letterhead)

	pdf.ln(20)
	write_subtitle(pdf, "1. Landscape Analysis")
	pdf.ln(20)
	pdf.image("images_summary_pdf_temp/landscape1.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(5)
	pdf.image("images_summary_pdf_temp/landscape2.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(5)
	pdf.image("images_summary_pdf_temp/landscape3.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(10)


	pdf.add_page()
	create_letterhead(pdf, WIDTH, letterhead)

	pdf.ln(20)
	write_subtitle(pdf, "2. Environment Analysis")
	pdf.ln(20)
	pdf.image("images_summary_pdf_temp/environment1.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(5)
	pdf.image("images_summary_pdf_temp/environment2.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(5)
	pdf.image("images_summary_pdf_temp/environment3.png", w=w, x=(WIDTH-w)/2)
	pdf.ln(10)

	pdf.add_page()
	create_letterhead(pdf, WIDTH, letterhead)

	pdf.ln(20)
	write_subtitle(pdf, "3. Duration Analysis")
	pdf.ln(20)
	pdf.image("images_summary_pdf_temp/duration1.png", w=w, x=(WIDTH-w)/2)


	pdf.output(file_name, 'F')
	
	shutil.rmtree('images_summary_pdf_temp')
	
	
