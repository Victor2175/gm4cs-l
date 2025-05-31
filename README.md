# gm4cs
Generative Models for Climate Science (Semester Project)

This repository contains code for the semester project on generative modeling of climate data.

## Project Structure
gm4cs/  
├── data/  
│   ├── ssp585.pkl

├── models/
│   ├── trend_vae_model.pt
│   ├── ssp585.pkl

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

gm4cs/
├── archive/ 

├── data/
│ └── ssp585_time_series.pkl 
├── models/
│ ├── trend_vae_model.pt 
│ └── linear_vae_model.pt 
├── outputs_ols/ 
├── outputs_rrr/ 
├── utils/ 
│ ├── animation.py
│ ├── data_loading.py
│ ├── data_processing.py
│ ├── metrics.py
│ ├── regression.py
│ ├── pipeline.py
│ └── trend_vae.py
├── RRR.ipynb
├── VAE.ipynb
├── Data_exploration.ipynb
├── train_and_generate.py 
├── vae.py 
├── Harrison_Global_Warming_Project.pdf
└── README.md
