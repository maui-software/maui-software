import numpy as np
import pandas as pd


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

def card_summary(df, show_plot=True):
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

def landscape_environment_heatmap(df, show_plot=True):
	df_group = df.groupby(['landscape', 'environment'], as_index=False)['file_path'].count()
	df_group = df_group.rename(columns={"file_path": "count"})
	
	
	df_group_temp = df_group.pivot(index='landscape', columns='environment', values='count')

	fig = px.imshow(df_group_temp, color_continuous_scale='Viridis', text_auto=True, title='Landscape vs Environment Heatmap')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return df_group, fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_histogram(df, show_plot=True):

	fig = px.histogram(df, x="landscape", color="environment", opacity=0.7, title='Ammount of samples by Landscape')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_duration(df, show_plot=True):

	fig = px.box(df, x="landscape", y="duration", title='Duration distribution by Landscape')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_landscape_daily_distribution(df, show_plot=True):

	fig = px.histogram(df, x="dt", color="landscape", opacity=0.7, title='Ammount of samples by Day and Landscape')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_histogram(df, show_plot=True):

	fig = px.histogram(df, x="environment", color="landscape", opacity=0.7, title='Ammount of samples by Environment')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()

	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_duration(df, show_plot=True):

	fig = px.box(df, x="environment", y="duration", title='Duration distribution by Environment')
	fig.update_layout(title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_environment_daily_distribution(df, show_plot=True):

	fig = px.histogram(df, x="dt", color="environment", opacity=0.7, title='Ammount of samples by Day and Environment')
	fig.update_layout(bargap=0.1, title_x=0.5)

	if show_plot:
		fig.show()
	
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

def plot_duration_distribution(df, show_plot=True):
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
	HEIGHT = hight

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

	intro_text = "This report contains a brief exploratory data analysis comprehending the data obtained by audio file names. The objective is to present an overview of the acoustic landscapes and environments of the recordings, as well as their duration. Further analysis such as false color spectrograms and acoustic indices summarization can be performed with Maui Sotware analysis and visualization tools."
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
	
	
