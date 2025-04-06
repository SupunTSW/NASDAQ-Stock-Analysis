Title: Predicting Financial Market Movements Using NASDAQ Historical Daily Prices Dataset 

This project focused on predicting stock price movements using multivariate time series data. Using the NASDAQ Historical Daily Prices dataset, the goal is to develop a deep learning model that can accurately forecast future stock prices, specifically the Close price.

Objective:
To develop a deep learning model capable of predicting NASDAQ stock prices using historical daily price data by:

  - Handling and preprocessing large-scale financial datasets
  - Engineering meaningful time-series features
  - Designing and training advanced neural network models (LSTM, GRU, etc.)
  - Evaluating model performance and optimizing for accuracy


How to Run the Project:

1. 📦 Clone the Repository

   
2. 🧪 Create and Activate a Virtual Environment

   
3. 📚 Install Dependencies using requirements.txt file
   ( Special Note: The ta-lib package was installed manually (not via pip), and may require manual installation depending on your system. It is used in the feature_engineering.py file to generate technical indicators for the stock price data.)


4. 📁 Prepare the Dataset
   - The raw dataset is stored in the data/raw folder.
   - Please use this raw data file as the input for data_preprocessing.py


5. 🧹 Run Preprocessing (data_preprocessing.py)
   - Begin by running the "preprocess_data()" function in the "data_preprocessing.py" script. Give the raw data file path to the function.
   - It generates three CSV files:
      - nasdaq_train_processed.csv
      - nasdaq_test_processed.csv
      - nasdaq_preprocessed_unscaled.csv
   - Among the three files generated, use the "nasdaq_preprocessed_unscaled.csv" file for feature engineering.


6. 🧹 Run Feature Engineering (feature_engineering.py)
   - Begin by running the "engineer_features()" function in the "feature_engineering.py" script. Give the "nasdaq_preprocessed_unscaled.csv" file path to the function.
   - It generates four CSV files:
      - feature_selection_results.csv
      - nasdaq_all_features.csv
      - nasdaq_selected_features_unscaled.csv
      - nasdaq_selected_features_scaled.csv
   - Among the four files generated, use the "nasdaq_selected_features_scaled.csv" file for model training and evaluation.
     (Note: The feature selection process takes approximately 3 to 4 minutes to complete.) 🧰
  
7. 🧠 Train the Models (train.py)
   - Specify the file path for the "nasdaq_selected_features_scaled.csv" dataset in the "main()" function within the "train.py" script.
   - Run the main() function
   - It creates a 'results' folder in the working directory and stores all visualization plots and trained models.
   - It also saves the best-performing Deep Learning model in the 'models' folder as 'trained_model.h5'.


8. 📈 Evaluate the Models
   - Evaluation plots and metrics (MAE, RMSE, etc.) will be generated during training and saved to the results/ directory.
  
