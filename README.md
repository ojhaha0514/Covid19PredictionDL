
# Enhancing COVID-19 Forecasts Through Multivariate Deep Learning Models

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-blue)
![COVID-19](https://img.shields.io/badge/COVID--19-Time%20Series%20Prediction-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Overview

This project explores how **multivariate deep learning (DL) models** can enhance COVID-19 forecasting by leveraging additional data from countries with similar infection trends. Using **Dynamic Time Warping (DTW)**, we rank countries by their similarity in infection trends and extend univariate DL models to multivariate ones, significantly improving prediction accuracy.

## Key Highlights

- **7 DL Models Evaluated**: Transformer, TCN, CNN-LSTM, BiLSTM, GRU, RNN, and LSTM.
- **Transformer Model Excellence**: Achieved an **average 60.15% reduction in RMSE** compared to univariate models.
- Integration of additional country data using **DTW rankings** for better performance under data-scarcity conditions.
- Statistically validated improvements via the **Diebold-Mariano test**.

## Methods

1. **Data Preprocessing**:
   - Dataset: COVID-19 time series data with corrections for new cases.
   - Filtered for countries with populations ≥10 million.
   - Time period: January 1, 2022 – July 31, 2022.

2. **Dynamic Time Warping (DTW)**:
   - Calculates similarity of infection trends between countries.
   - Generates rankings for integrating top similar countries into multivariate models.

3. **Modeling**:
   - Extended univariate DL models to multivariate ones using data from top-ranked countries.
   - Implemented models include:
     - Transformer
     - Temporal Convolutional Network (TCN)
     - CNN-LSTM
     - BiLSTM
     - GRU
     - RNN
     - LSTM

4. **Validation**:
   - Evaluation metric: Root Mean Square Error (RMSE).
   - Timeframes: Five target countries and five forecast periods.

## Results

- **Best Performer**: Multivariate **Transformer** model.
  - 60.15% reduction in mean RMSE across test cases.
- Other models' RMSE improvements:
  - TCN: 36.28%
  - CNN-LSTM: 29.47%
  - BiLSTM: 21.07%
  - GRU: 21.43%
  - RNN: 17.46%
  - LSTM: 16.38%

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ojhaha0514/Covid19PredictionDL.git
   cd Covid19PredictionCompartment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts:
   - **DTW ranking**: `code_DTW_rank.py`
   - **Model tuning**: `code_tuning_DL.py`
   - **Multivariate DL models**: `code_multivariate_DL.py`

---

## Usage

### Dynamic Time Warping (DTW)
Calculate similarity rankings of infection trends:
```bash
python code_DTW_rank.py
```

### Hyperparameter Tuning
Perform automated tuning for DL models:
```bash
python code_tuning_DL.py
```

### Train Deep Learning Models
Train multivariate models with additional country data:
```bash
python code_multivariate_DL.py
```



---

## File Structure

```
.
├── code_DTW_rank.py        # DTW-based country ranking
├── code_tuning_DL.py       # Hyperparameter tuning
├── code_multivariate_DL.py # Deep Learning model training
├── owid-covid-data.csv     # Preprocessed COVID-19 dataset
└── results/                # Generated results and logs
```

---

## Authors

- **Jooha Oh**<sup>1*</sup>, **Zhe Liu**<sup>2*</sup>, **Kyulhee Han**<sup>2</sup>, **Taewan Goo**<sup>2</sup>, **Hanbyul Song**<sup>2</sup>, **Jiwon Park**<sup>2</sup>
- **Corresponding Author**<sup>1</sup>: [Taesung Park](mailto:tspark@stats.snu.ac.kr)
- <sup>1</sup>Department of Statistics, Seoul National University, Seoul, Republic of Korea
- <sup>2</sup>Interdisciplinary Program in Bioinformatics, Seoul National University, Seoul, Republic of Korea
- *These authors contributed equally to this work.



---

## Acknowledgments

- The code utilizes **PyTorch** and **Ray Tune** for model development and tuning.
- Publicly available data collected by Our World In Data were used in this study and can be found at following repository: [https://covid.ourworldindata.org]{https://covid.ourworldindata.org}.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
