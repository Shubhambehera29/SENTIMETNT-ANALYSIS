# Multi-Domain Sentiment Analysis using NLP and Machine Learning

## Overview

This project develops a machine-learning-based sentiment analysis system capable of classifying text reviews as **Positive** or **Negative**.

The model is trained using reviews from multiple domains, including:

* Movie reviews
* Restaurant reviews
* Product reviews
* Social-media-style text

The project demonstrates how Natural Language Processing and machine learning can be combined to analyze opinions across different domains.

## Objectives

* Perform text preprocessing and cleaning.
* Combine multiple sentiment-analysis datasets.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression classification model.
* Evaluate the model using standard classification metrics.
* Test the model using custom out-of-sample reviews.
* Predict sentiment and confidence for user-entered text.

## Datasets

The project uses the following publicly available datasets:

* **IMDb Movie Reviews**
* **Yelp Polarity**
* **Amazon Polarity**
* **Sentiment140** *(optional, if included in the notebook)*

The datasets are loaded or downloaded through the notebook. They are not directly included in this repository because of their large size.

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* NLTK-style text preprocessing concepts
* TF-IDF Vectorization
* Logistic Regression
* Matplotlib
* Seaborn
* Joblib

## Methodology

The overall workflow is:

1. Load multiple sentiment datasets.
2. Clean and normalize the review text.
3. Balance the positive and negative samples.
4. Combine the datasets.
5. Split the data into training and testing sets.
6. Apply TF-IDF vectorization using unigrams and bigrams.
7. Train a Logistic Regression classifier.
8. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
9. Test the model on custom reviews.
10. Use an interactive prediction loop for new user input.

## Model

The primary model uses:

* **Feature extraction:** TF-IDF
* **N-gram range:** Unigrams and bigrams
* **Classifier:** Logistic Regression

The model predicts one of two classes:

* `Positive`
* `Negative`

The confidence score represents the probability estimated by the classifier. It should not be interpreted as a guarantee of prediction correctness.

## Example Prediction

```text
Enter a review: The product quality is excellent and delivery was fast.

Sentiment: Positive
Confidence: 98.XX%
```

## Mixed-Reaction Reviews

The project also tests challenging reviews containing both positive and negative opinions, such as:

> The camera quality is excellent, but the battery drains very quickly.

Since the current model is trained for binary classification, these reviews are classified as either Positive or Negative rather than as a separate Mixed class.

## Repository Contents

Multi-Domain-Sentiment-Analysis/
│
├── Multi_Domain_Sentiment_Analysis.ipynb
├── README.md
└── sentiment_analysis_model.pkl   # Optional


## How to Run

1. Open the `.ipynb` file in Google Colab.
2. Run the installation and dataset-loading cells.
3. Execute the preprocessing and model-training cells.
4. Evaluate the model.
5. Run the custom prediction section.
6. Enter your own review in the interactive prediction loop.

## Future Improvements

* Add a separate **Mixed** sentiment class.
* Use transformer-based models such as BERT or RoBERTa.
* Improve handling of sarcasm and negation.
* Perform domain-wise error analysis.
* Deploy the model using Streamlit or Flask.
* Add multilingual sentiment analysis.

## Author

**Shubham Behera**

B.Tech — Computer Science and Engineering, Data Science
C.V. Raman Global University, Bhubaneswar
