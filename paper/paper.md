---
title: 'Maui: A Python Visualization Framework for Ecoacoustics Data'
tags:
  - Python
  - soundscape data analytics
  - ecoacoustics data
  - data visualization
authors:
  - name: Caio Ferreira Bernardo
    orcid: 0009-0002-9447-8576
    equal-contrib: true
    affiliation:  1
  - name: Maria Cristina Ferreira de Oliveira
    orcid: 0000-0002-4729-5104
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1

affiliations:
 - name: Instituto de Ciências Matemáticas e de Computação - Universidade de São Paulo (USP), São Carlos, Brazil
   index: 1
date: 10 September 2025
bibliography: paper.bib

---

# Summary

Passive Acoustic Monitoring (PAM) generates vast volumes of environmental audio recordings. 
There is a diverse range of tools to study this data, either to compute mathematical features 
(acoustic indices) or to perform machine learning tasks such as classification. However, 
researchers still lack unified tools for visually exploring soundscape repositories.
We introduce Maui, an open‑source Python framework designed to simplify exploratory analysis 
of large ecoacoustic datasets by placing visualization directly into the PAM workflow.
Maui provides modules for data ingestion and metadata parsing, supports flexible incorporation 
of user‑computed acoustic indices, and offers a suite of high‑level plotting functions. 
It includes methods for creating false‑color spectrograms, customizable Diel plots, and 
radar and violin plots for multivariate index comparison. These visualizations are useful 
to reveal temporal, spatial, and taxonomic patterns at scale. With a modular architecture, 
Maui leverages existing Python libraries for bioacoustic data processing, while focusing 
on interactive, publication‑quality graphics that facilitate hypothesis generation and 
large‑scale soundscape synthesis. By filling a gap in ecoacoustic visual data analytics, 
Maui helps researchers to study soundscape dynamics.

# Statement of need

The sounds originating from anthrophonic, biophonic, and geophonic sources in a landscape define its soundscape[@pijanowski2011soundscape]. Substantial amounts of acoustic data, particularly from biophonic sources, can be captured using low-cost autonomous recorders deployed for Passive Acoustic Monitoring (PAM)[@browning2017passive]. An active research line in environmental ecology, acoustic ecology addresses the study of natural soundscapes [@grinfeder2022we]. The discipline relies heavily on computational methods for audio data processing and analysis [@Pijanowski2024; @Napier2024ESWA].

A diversity of PAM data processing pipelines is described in the literature. In the particular context of detecting the presence of animal species in the recordings, [@gibb2019emerging] defines a seven-step pipeline (see Figure \ref{fig:proposed_pam_pipeline}). Departing from data acquisition, sampling the recordings before analysis is often necessary, given the large data volumes collected. Researchers extract the associated metadata from the resulting subset of audio files, including location, time, climate conditions, and recorder type. A fourth step involves preprocessing the audio files, e.g., to reduce noise and emphasize relevant signals. Researchers can then perform acoustic event detection and labeling, e.g., to identify animal species. This typically involves multiple iterations of computing metrics such as acoustic features, ecological indices, and conducting statistical analyses of intermediate results to gain a comprehensive understanding of the soundscape.


As PAM pipelines are instantiated multiple times (see Figure~\ref{fig:proposed_pam_pipeline}), researchers typically accumulate vast datasets of audio recordings collected at multiple sites over extended periods. These large repositories of environmental acoustic recordings are a valuable source of knowledge when analyzed from a global perspective. Knowledge extraction requires practical software tools and libraries to streamline data exploration and analysis. Analysts need flexibility to investigate their accumulated data from multiple perspectives, considering different data and metadata properties. Conducting global investigations can uncover insights beyond previous soundscape analyses, help identify potential improvements in existing practices and methodologies, and support large-scale PAM data analysis in the long term [@Napier2024ESWA]. 

Data visualization is a powerful tool for extracting insights in this inherently exploratory context. It enables the representation of metadata and acoustic features across time and locations, providing comprehensive overviews of the data repositories from multiple perspectives. Compelling visualizations can enhance data exploration and summarization beyond the standard processing pipeline. We consider an extended PAM pipeline that integrates visualization into a framework to promote exploratory analysis of the data accumulated in acoustic repositories resulting from multiple instantiations of the standard pipeline, as illustrated in Figure \ref{fig:proposed_pam_pipeline}. 
The dissemination of acoustic ecology practices has motivated many open-source tools and libraries to facilitate tasks such as computing acoustic features and applying machine learning algorithms. Following this trend, we introduce Maui, a Python package to support visual exploratory tasks on ecoacoustic data repositories. 


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References