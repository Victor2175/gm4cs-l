# gm4cs
Generative models for Climate science (semester project)

This repo will enable to share the code of the semester project.

## Project Structure
( ``` )
gm4cs-1/
├── archive/                # Old notebooks and code versions
├── data/
│   └── ssp585_time_series.pkl  # Main dataset (climate time series)
├── models/
│   ├── trend_vae_model.pt        # VAE model for trend generation
│   └── linear_vae_model.pt       # Trained VAE weights
├── outputs_ols/            # Outputs from OLS regression
├── outputs_rrr/            # Outputs from Reduced Rank Regression (RRR)
├── utils/                  # Utility scripts
│   ├── animation.py
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── metrics.py
│   ├── regression.py
│   ├── pipeline.py
│   └── trend_vae.py
├── RRR.ipynb               # Tutorial for Reduced Rank Regression
├── VAE.ipynb               # Tutorial for VAE training and visualization
├── Data_exploration.ipynb  # Data exploration notebook
├── train_and_generate.py   # Main training script for RRR
├── vae.py                  # Script to train the VAE
├── Harrison_Global_Warming_Project.pdf # Project report
└── README.md               # Project documentation
( ``` )

To run the cross-validation with the VAE model run: 
python vae.py

To run the cross-validation for the RRR model and generate outputs:
python -u train_and_generate.py \
  --data_path ./data \
  --filename ssp585_time_series.pkl \
  --output_dir ./outputs_rrr_2 \
  --all_models \
  --center \
  --num_runs 4 \
  --mean_weight 0.75 \
  --variance_weight 0.25 \
  --option 1 \
  --normalise \
  > logs_rrr_2.out 2>&1 &