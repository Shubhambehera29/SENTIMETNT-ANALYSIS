# Multi-Domain Sentiment Analysis using NLP and Machine Learning

## Overview

This project focuses on developing a machine-learning-based sentiment analysis system that classifies text reviews as **Positive** or **Negative**.

The project was developed as part of my learning journey in Python, Data Science, and Natural Language Processing. It helped me gain practical experience in dataset integration, text preprocessing, feature extraction, model training, evaluation, and real-time sentiment prediction.

The model is trained using sentiment datasets from multiple domains, including movie reviews, restaurant reviews, product reviews, and social-media text.

## Objectives

* Understand the fundamentals of Natural Language Processing.
* Combine sentiment datasets from multiple domains.
* Perform text cleaning and preprocessing.
* Convert text into numerical features using TF-IDF.
* Train a machine-learning classification model.
* Evaluate the model using standard performance metrics.
* Analyze performance across different datasets.
* Test the model on custom and mixed-reaction reviews.
* Implement an interactive sentiment prediction system.

## Datasets Used

The project uses four publicly available sentiment datasets:

| Dataset         | Domain                         | Samples Used |
| --------------- | ------------------------------ | -----------: |
| IMDb            | Movie reviews                  |       20,000 |
| Yelp Polarity   | Restaurant and service reviews |       20,000 |
| Amazon Polarity | Product reviews                |       20,000 |
| Sentiment140    | Social-media text              |       20,000 |
| **Total**       | **Multiple domains**           |   **80,000** |

For each dataset, balanced samples were selected to maintain an equal number of positive and negative reviews.

### Dataset Distribution

* **Positive reviews:** 40,000
* **Negative reviews:** 40,000
* **Total reviews:** 80,000

The final dataset was divided into training and testing subsets using an 80:20 split.

| Data Split   | Number of Reviews |
| ------------ | ----------------: |
| Training set |            64,000 |
| Testing set  |            16,000 |
| **Total**    |        **80,000** |

## Data Preprocessing

The following preprocessing steps were applied:

1. Converted text to lowercase.
2. Removed HTML tags.
3. Removed URLs.
4. Normalized extra whitespace.
5. Combined relevant text fields, such as product titles and review content.
6. Balanced positive and negative samples from each dataset.
7. Combined the processed datasets into a single multi-domain dataset.

## Methodology

The overall workflow of the project is:

Multiple Sentiment Datasets
            ↓
Data Loading and Sampling
            ↓
Text Cleaning and Preprocessing
            ↓
Dataset Combination
            ↓
Train-Test Split
            ↓
TF-IDF Feature Extraction
            ↓
Logistic Regression Model
            ↓
Model Evaluation
            ↓
Custom Review Prediction


## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Matplotlib
* Seaborn
* Joblib

## Machine Learning Model

The primary classification pipeline consists of:

### Feature Extraction

**TF-IDF Vectorization** was used to convert text into numerical feature vectors.

The model used:

* Unigrams and bigrams
* Maximum features: 100,000
* Minimum document frequency: 2
* Sublinear TF scaling

### Classifier

**Logistic Regression** was used as the classification algorithm.

The model predicts two sentiment classes:

* `0 → Negative`
* `1 → Positive`

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Dataset-wise accuracy

### Overall Classification Report

| Class            | Precision | Recall | F1-Score |    Support |
| ---------------- | --------: | -----: | -------: | ---------: |
| Negative         |      0.87 |   0.86 |     0.87 |      8,000 |
| Positive         |      0.86 |   0.87 |     0.87 |      8,000 |
| **Accuracy**     |           |        | **0.87** | **16,000** |
| Macro Average    |      0.87 |   0.87 |     0.87 |     16,000 |
| Weighted Average |      0.87 |   0.87 |     0.87 |     16,000 |

The model achieved an overall accuracy of approximately **86.7%**, which is reported as **87%** in the classification report.

### Confusion Matrix

The confusion matrix produced the following results:

| Actual / Predicted | Negative | Positive |
| ------------------ | -------: | -------: |
| **Negative**       |    6,913 |    1,087 |
| **Positive**       |    1,047 |    6,953 |

Where:

* **True Negatives:** 6,913
* **False Positives:** 1,087
* **False Negatives:** 1,047
* **True Positives:** 6,953

The confusion matrix shows that the model performed relatively consistently in identifying both positive and negative reviews.

## Performance Across Domains

The model was also evaluated separately on selected datasets to understand how well it performs across different domains.

| Dataset | Accuracy |
| ------- | -------: |
| IMDb    |   88.52% |
| Yelp    |   92.42% |
| Amazon  |   89.74% |

### Observations

* The model achieved its highest accuracy on the **Yelp dataset**, with **92.42%**.
* The model achieved **89.74%** accuracy on Amazon product reviews.
* The model achieved **88.52%** accuracy on IMDb movie reviews.
* Performance varies across domains because vocabulary, writing style, review length, and contextual expressions differ between datasets.

## Custom and Mixed-Reaction Reviews

In addition to evaluating the model on dataset samples, custom out-of-sample reviews were used to test its behaviour on unseen text.

The testing examples included:

* Clearly positive reviews
* Clearly negative reviews
* Mixed-reaction reviews containing both positive and negative opinions
* Challenging reviews involving negation and contrasting statements

For example:

> The camera quality is excellent, but the battery drains very quickly.

The current model uses binary classification, so it predicts either Positive or Negative. It does not currently have a separate Mixed sentiment class.

## Interactive Prediction

The project includes an interactive prediction loop that allows users to enter their own reviews and receive a predicted sentiment and confidence score.

Example:

```text
Enter a review: The product quality is excellent and delivery was fast.

Sentiment: Positive
Confidence: 98.XX%
```

The confidence value is the probability estimated by the classifier and should not be considered a guarantee of correctness.

## How to Run the Project

1. Open the notebook in Google Colab.
2. Run the installation cells.
3. Download or load the required datasets.
4. Execute the preprocessing cells.
5. Create the combined dataset.
6. Train the TF-IDF and Logistic Regression pipeline.
7. Run the evaluation cells.
8. Test the model using custom reviews.
9. Use the interactive prediction loop for new text.

## Repository Contents

```text
SENTIMETNT-ANALYSIS/
│
├── Multi_Domain_Sentiment_Analysis.ipynb
├── README.md
└── sentiment_analysis_model.pkl
```

> The model file is optional and may not be included if it is too large. The notebook contains the complete training pipeline for recreating the model.

## Key Learning Outcomes

Through this project, I gained practical experience in:

* Working with multiple real-world datasets.
* Handling and balancing large text datasets.
* Applying NLP preprocessing techniques.
* Using TF-IDF for text representation.
* Training and evaluating a classification model.
* Understanding confusion matrices and classification metrics.
* Performing domain-wise performance analysis.
* Testing machine-learning models on unseen and mixed-polarity text.

## Future Improvements

* Introduce a separate **Mixed** sentiment class.
* Explore transformer-based models such as BERT and RoBERTa.
* Improve handling of sarcasm, negation, and contextual sentiment.
* Perform detailed error analysis across domains.
* Add multilingual sentiment analysis.
* Deploy the model using Streamlit or Flask.
* Compare Logistic Regression with Naive Bayes, Linear SVM, and transformer models.

## Author

**Shubham Behera**

B.Tech — Computer Science and Engineering
C.V. Raman Global University, Bhubaneswar

