# Credit Card Fraud Detection System

<!-- ![Fraud Detection Demo](demo.gif) <!-- Add a demo gif if available -->
#Credit Card Fraud Detection Using State-of-Art Machine and Deeping Learning Algorithms
A Model for detecting fraudulent credit card transactions using multiple algorithms including KNN, Decision Trees, Logistic Regression, Random Forest, and CNN.

## Features

- Real-time fraud prediction
- Multiple model comparison (5 different algorithms)
- Interactive web interface
- Demo data generation
- Visual results display
- Comprehensive feature engineering

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**:
  - Scikit-learn (v1.6.1)
  - TensorFlow (v2.8.0)
  - XGBoost
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection   
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Train the models:
   ```bash
   python train_models.py
5. Start the Flask server:
   ```bash
   python app.py
  Access the web interface at:
  ```bash
  http://localhost:5000
  
