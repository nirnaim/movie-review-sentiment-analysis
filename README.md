# Movie Review Sentiment Analysis

## Overview

This project demonstrates a complete machine learning pipeline for sentiment analysis on movie reviews. Using the IMDb Movie Reviews dataset, we classify reviews as either positive or negative. The process includes downloading and preprocessing the data, extracting features using TF-IDF, training a logistic regression model, and evaluating its performance. This project highlights key steps in natural language processing (NLP) and machine learning, showcasing data handling, model training, and visualization techniques.

## Table of Contents

- [Installation](#installation)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the Repository**: Download the project files from the repository.
2. **Create and Activate a Virtual Environment**: Set up a virtual environment to manage dependencies.
3. **Install Required Libraries**: Install the necessary Python libraries including `numpy`, `pandas`, `scikit-learn`, `nltk`, `matplotlib`, and `seaborn`.

## Data Collection and Preprocessing

1. **Download the IMDb Movie Reviews Dataset**: Obtain the dataset which contains labeled movie reviews for training and testing.
2. **Load the Data**: Read the data files into a Pandas DataFrame.
3. **Preprocess the Data**: 
   - Convert text to lowercase.
   - Remove punctuation and stopwords.
   - Tokenize the text.
   - Save the cleaned data for further processing.

## Model Training

1. **Extract Features**: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
2. **Train the Model**: 
   - Split the dataset into training and testing sets.
   - Train a logistic regression model on the TF-IDF features.
   - Save the trained model and the TF-IDF vectorizer for future use.

## Model Evaluation

1. **Evaluate the Model**: 
   - Predict the sentiments for the test data.
   - Calculate the accuracy and generate a classification report.
   - Display a confusion matrix to visualize the model's performance.
2. **Visualize the Results**: Create plots to help interpret the model's performance metrics.

## Usage

1. **Activate the Virtual Environment**: Ensure that the virtual environment is active.
2. **Follow the Steps**: 
   - Preprocess the data.
   - Train the model.
   - Evaluate the model's performance.
3. **Modify and Expand**: Feel free to enhance the project by trying different models, preprocessing techniques, or additional data.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
