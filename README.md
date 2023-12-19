# Domain Agnostic Sentiment Analysis
## Overview
"Domain-Agnostic Sentiment Analysis" is a project aimed at developing a versatile sentiment analysis model capable of accurately predicting sentiments across various domains. Initially utilizing a dataset of Amazon reviews, the project's ultimate goal is to integrate diverse datasets, enhancing its domain-agnostic capabilities. The project encompasses a detailed data processing pipeline, model comparison, hyperparameter tuning, and a user-friendly Flask web application for real-time sentiment prediction.
## Data Source
The initial phase uses the Amazon Reviews dataset from Kaggle: [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). This dataset comprises Amazon reviews labeled as positive or negative. Future iterations will incorporate datasets from multiple domains to strengthen the model's domain-agnostic attribute.
## Key Features
- **Data Pre-processing/Cleaning:** Initial steps include cleaning and preparing the data for analysis.
- **Feature Engineering:** Techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and Count Vectorization are employed to extract features from the text data.
- **Model Training and Comparison:** The project explores a range of machine learning models:
  - Random Forest
  - Gradient Boosting
  - Naive Bayes
  - Support Vector Machines
  - Logistic Regression
  - LSTM (Long Short-Term Memory) neural network, with and without FastText word embeddings.
- **Hyperparameter Tuning:** Utilization of Grid Search for optimizing model parameters.
- **Performance Evaluation:** Models are validated (n-cross validation), evaluated and compared based on precision, recall, and accuracy metrics.
## Best Performing Model
After rigorous testing and evaluation, the best performing model is identified. Its optimized version is serialized into a pickle file for use in real-time predictions.
## Web Application
- **Framework:** Flask
- **Functionality:** The application allows users to input text, which is then processed by the pickled model to predict the sentiment of the input.
- **User Interface:** Designed to be user-friendly, facilitating easy input of text and displaying the sentiment analysis results.
## Getting Started
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Flask
- Necessary Python libraries (listed in requirements.txt)
### Installation
- Clone the repository.
- Install the required Python packages: `pip install -r requirements.txt`
- Run the Jupyter notebook Domain-Agnostic-Sentiment-Analysis.ipynb to understand the model development process.
- Launch the Flask app for real-time sentiment analysis.
### Usage
- It is advised to create a virtual environment. Refer the [link](https://docs.python.org/3/library/venv.html) for more details.
- To use the web application, run the Flask server using `flask run` and navigate to the provided local URL.
- Input the text for sentiment analysis in the UI and submit to receive the sentiment prediction.
### UI Screenshots
<img width="508" alt="Screen Shot 2023-12-19 at 6 45 38 PM" src="https://github.com/gkseehra/Domain-Agnostic-Sentiment-Analysis/assets/35463826/66775f47-c5ce-467e-9054-0abbb6c30d54">
<img width="498" alt="Screen Shot 2023-12-19 at 6 46 04 PM" src="https://github.com/gkseehra/Domain-Agnostic-Sentiment-Analysis/assets/35463826/388ffc6f-92e0-4d60-84e4-fce7c115f25e">

## Contact
For any queries or suggestions, please reach out to gurleenkseehra@gmail.com
