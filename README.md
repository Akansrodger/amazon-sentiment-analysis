
<img width="640" height="480" alt="WORDCLOUD" src="https://github.com/user-attachments/assets/fe0f578e-e329-464e-b526-32f5d7fe98c1" />

# amazon-sentiment-analysis
**Tech Stack**

   Python | Scikit-learn | NLTK | Pandas | Matplotlib | Seaborn|re(Regular expression)


**⚙️ How to Run**
git clone https://github.com/yourname/repo
pip install -r requirements.txt
python main.py

**Sentiment Analysis of Amazon Product Reviews**
**📖 Overview**

This project builds a machine learning model to classify Amazon product reviews as Positive or Negative using Natural Language Processing (NLP) techniques.

The goal was to:

Clean and preprocess raw text data

Convert text into numerical features using TF-IDF

Train and compare multiple machine learning models

Evaluate performance using robust metrics

Visualize sentiment distribution and important features

Perform real-time sentiment prediction

**📊 Dataset**

 Dataset: Amazon Review Polarity Dataset
  Download from: https://www.kaggle.com/..
Training set: 200,000 sampled reviews

Test set: 50,000 sampled reviews

Balanced classes:

Positive (1)

Negative (0)

**🧹 Data Preprocessing**
Lowercasing text

Removing URLs

Removing punctuation

Removing numbers

Removing stopwords (NLTK)

Combining title + review text

**Feature Engineering**

TF-IDF Vectorization:

max_features=5000

min_df=5

max_df=0.9

This converts text into numerical vectors for machine learning.

**Models Compared**

Logistic Regression (primary model)

Multinomial Naive Bayes

Linear Support Vector Machine (SVM)

**📈 Results**

Accuracy: ~89%

Precision: 0.89

Recall: 0.89

F1-score: 0.89

Balanced performance across both classes

Confusion Matrix shows symmetric error distribution.

**🔍 Key Insights**

Dataset is perfectly balanced.

Model performs equally well on positive and negative reviews.

Most influential positive words include: "great", "love", "best"

Most influential negative words include: "worst", "waste", "poor"

**🧠 Real-Time Prediction**

The model can classify new reviews dynamically:

Example:

"This product is amazing!" → Positive
"Waste of money." → Negative

**Visualizations Included**

Sentiment distribution bar chart

Confusion matrix

Word cloud for positive reviews

**Future Improvements**

Hyperparameter tuning

Cross-validation

Transformer-based models (BERT)

Deploy as web app using Streamlit
