# Spire's Classification with Automated Semi-supervised Algorithms (CASA) package

## Overview

Spire's `CASA` is a Python package to perform unsupervised and semi-supervised machine learning (ML) classification algorithms on generic tensors of pre-processed data, such as time series, altitude profiles, images, DDMs and spectra. Mainly tested on Earth Information (EI) data produced by our own satellites, such as GNSS-RO sTEC profiles and GNSS-R DDMs. It produces a database of labeled clusters that can be used to classify new unlabeled data.

It includes the following blocks:

* Compute wavelet spectra of 1-D data tensors (e.g., time series, altitude profiles)
* Downsampling of input 2-D data tensors (e.g., images, DDMs)
* Parallelized distance matrix computation using earth mover's distance (EMD, aka Wasserstein metric)
* Spetral clustering using precomputed distance matrix
* Self-tuned spectral clustering using precomputed distance matrix
* HDBSCAN clustering using precomputed distance matrix
* Classification of new data based on database of labeled clusters

## Prerequisites

`CASA` requires the following packages for installation:

* numpy
* pandas
* scipy
* sklearn
* multiprocessing
* hdbscan (if HDNSCAN clustering is required)
* seaborn

## Installation

    pip install casa


### Authors

- Giorgio Savastano (<giorgiosavastano@gmail.com>)
- Karl Nordstrom (<karl.nordstrom@spire.com>)

## References

Savastano, G., K. Nordström, and M. J. Angling (2022), Semi-supervised Classification of Lower-Ionospheric Perturbations using GNSS Radio Occultation Observations from Spire’s Cubesat Constellation. Submitt. to JSWSC.
