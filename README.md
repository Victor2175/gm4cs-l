# gm4cs
Generative Models for Climate Science (Semester Project)

This repository contains code for the semester project on generative modeling of climate data.

## Project Structure

gm4cs/  
├── archive/  # Old notebooks and code versions
│   ├── .h5 files (experiment-specific target data)  
├── LFS/  
│   ├── .h5 files (experiment-specific input data)  

notebooks/  
├── Jupyter notebooks for testing and analysis, including results visualization and evaluation metrics.  

inference_time_plots/  
├── Contains plots with the evaluation of inference times for both the baseline and Bayesian network

models/  
├── Saved the 2 best models obtained after hyperparameter tuning. 

pkl_files/  
├── Pickle files for saving/loading data.

scalers/  
├── Scalers used for normalizing or standardizing the data.  

main.py  
├── Python script to launch training for any specified model.  

README.md  
requirements.txt  
