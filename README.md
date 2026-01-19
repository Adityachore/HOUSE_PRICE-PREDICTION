# House Price Prediction AI

An intelligent property valuation system that uses machine learning to predict house prices based on various features such as living area, bedrooms, bathrooms, location type, and exterior quality.

## Features
- **Predictive Valuation**: Real-time property price estimation using a Linear Regression model.
- **Dynamic Frontend**: Modern, responsive UI built with HTML/CSS and Lucide icons.
- **Flask Backend**: Robust API serving model predictions.
- **Automated Training**: Includes a training script (`ml.py`) to retrain the model on new data.

## Project Structure
- `app.py`: Main Flask application.
- `ml.py`: Machine learning training script.
- `house_price_model.pkl`: Trained model file.
- `model_columns.pkl`: Metadata for model columns.
- `templates/`: HTML templates for the frontend.
- `static/`: Static assets (images, styles).
- `train (3).csv`: Dataset used for training.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Adityachore/HOUSE_PRICE-PREDICTION.git
   cd HOUSE_PRICE-PREDICTION
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000`.

## How to Train the Model
If you want to retrain the model with updated data:
1. Ensure your dataset is named `train (3).csv` (or update the path in `ml.py`).
2. Run the training script:
   ```bash
   python ml.py
   ```

## Technologies Used
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, Lucide Icons
