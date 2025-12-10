# Stock-Price-prediction
📌 Overview

This project builds a deep learning model using LSTM (Long Short-Term Memory) networks to predict the next-day closing price of Tesla (TSLA) stock.
The model is trained on historical stock data and compared against a classical machine learning baseline (Random Forest).

The project includes:

Data preprocessing

Sliding window sequence creation

LSTM model training

Model evaluation (RMSE)

Visualization of actual vs. predicted prices

📂 Project Structure
├── train_tsla_keras.py           # Main LSTM training script
├── TSLA.csv                      # Historical TSLA stock price dataset
└── README.md                     # Project documentation

🧠 Model Summary
Model Architecture

LSTM layer (50 units, return sequences)

LSTM layer (50 units)

Dense output layer (1 neuron)

Optimizer: Adam

Loss function: MSE

Epochs: 50

📊 Dataset

The project uses TSLA.csv, containing:

Date

Open

High

Low

Close

Adj Close

Volume

Only the Close price is used for prediction.

Preprocessing Includes:

Sorting by date

MinMax scaling

60-day sliding window

80/20 train-test split
Batch size: 32

🚀 How to Run the Project
1. Install Requirements

Run the following:

pip install --upgrade pip
pip install tensorflow pandas numpy scikit-learn matplotlib joblib

2. Run the Training Script

Place TSLA.csv in the same folder, then run:

python train_tsla_keras.py


The script will:

Prepare data

Train the LSTM

Evaluate performance

Show prediction plots

Save model outputs

📈 Results

The trained LSTM model captures stock trends very well and produces smooth, realistic predictions compared to classical ML models.

Visualizations include:

Actual vs. predicted stock price plot

Training loss curves

🔧 Tools & Libraries Used

TensorFlow / Keras

Pandas

NumPy

Scikit-learn

Matplotlib

Joblib

