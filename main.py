
import pandas as pd

# Load training data
train_df = pd.read_csv("Data/train.csv", header=None)

# Load test data
test_df = pd.read_csv("Data/test.csv", header=None)

train_df = train_df.sample(200000, random_state=42)
test_df = test_df.sample(50000, random_state=42)

# Rename columns for clarity
train_df.columns = ["label", "title", "review"]
test_df.columns = ["label", "title", "review"]

print(train_df.head())
print(train_df.info())
print(test_df.info())

# Combining title + Review
train_df["text"] = train_df["title"] + " " + train_df["review"]
test_df["text"] = test_df["title"] + " " + test_df["review"]

#Now to clean the dataset
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)
# apply the function created to the dataset
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)
#changing the labels to a preferebly suit ML
train_df["label"] = train_df["label"] - 1
test_df["label"] = test_df["label"] - 1

#Now we can vectorize the text data using TF-IDF, which stands for Term Frequency-Inverse Document Frequency. This technique converts the text into numerical features that can be used by machine learning algorithms. It gives more weight to words that are unique to a document and less weight to common words across all documents.
from sklearn.feature_extraction.text import TfidfVectorizer

#The TfidfVectorizer is configured with a maximum of 3000 features, a minimum document frequency of 5, and a maximum document frequency of 0.9. This means that only the top 3000 most important words will be considered, and words that appear in less than 5 documents or more than 90% of the documents will be ignored. This helps to reduce noise and focus on the most relevant features for sentiment analysis.
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.9
)
#The TF-IDF vectors are created for both the training and test datasets. The fit_transform method is used on the training data to learn the vocabulary and create the TF-IDF matrix, while the transform method is used on the test data to create the TF-IDF matrix based on the learned vocabulary from the training data. This ensures that the same features are used for both datasets.
X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])
#The labels are extracted from the "label" column of the training and test datasets. The labels indicate whether a review is positive (1) or negative (0), which will be used as the target variable for training the machine learning model.
y_train = train_df["label"]
y_test = test_df["label"]

#Training the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Logistic Regression is a popular machine learning algorithm used for binary classification tasks, such as sentiment analysis. It models the relationship between the input features (in this case, the TF-IDF vectors) and the binary target variable (positive or negative sentiment) by estimating the probabilities of the classes. The "saga" solver is an optimization algorithm that is efficient for large datasets and supports L1 regularization, which can help with feature selection by shrinking less important feature coefficients to zero.
model = LogisticRegression(max_iter=1000, solver="saga")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

#Comparing the models, here i want to look at the best mathemathical approach that captures sentiment patterns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

# SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))    
# Now we can use the best model to predict sentiment on new reviews
#Now this is to test if the model works an how it identifies positive and negative reviews:
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

reviews = [
    "This product is amazing!",
    "Completely useless and waste of money",
    "It is okay, not bad",
    "I love it, highly recommend!",
    "Worst experience ever, do not buy!"

    ]

for r in reviews:
    print(r, "->", predict_sentiment(r))


#The classification report provides a detailed analysis of the model's performance, including precision, recall, f1-score, and support for each class. Precision is the ratio of true positives to the sum of true positives and false positives, indicating how many of the predicted positive instances are actually positive. Recall is the ratio of true positives to the sum of true positives and false negatives, indicating how many of the actual positive instances were correctly identified by the model. The f1-score is the harmonic mean of precision and recall, providing a single metric that balances both. Support refers to the number of actual occurrences of each class in the dataset.
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
#The value counts of the labels in the training and test datasets show the distribution of positive and negative reviews. This is important to understand if the dataset is balanced or imbalanced, which can affect the performance of the model. A balanced dataset has an equal number of positive and negative reviews, while an imbalanced dataset may have a majority of one class, which can lead to biased predictions.
print(y_train.value_counts())
print(y_test.value_counts())

#The full dataset distribution of the labels in the training dataset shows how many positive and negative reviews are present. The percentages provide a clearer understanding of the proportion of each class in the dataset, which is crucial for evaluating the model's performance and ensuring that it is not biased towards one class due to an imbalanced dataset.
print("Full dataset distribution:")
print(train_df['label'].value_counts())
print("\nAs percentages:")
print(train_df['label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')


#Now we can look at the most important features (words) that contribute to the sentiment classification
import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive = np.argsort(coefficients)[-20:]
top_negative = np.argsort(coefficients)[:20]

print("Top Positive Words:")
for index in top_positive:
    print(feature_names[index])

print("\nTop Negative Words:")
for index in top_negative:
    print(feature_names[index])
#The sentiment distribution plot shows the count of positive and negative reviews in the training dataset. This visualization helps to understand the balance of the dataset, which is important for training a machine learning model. If the dataset is imbalanced (e.g., significantly more positive reviews than negative reviews), it may lead to biased predictions, where the model may favor the majority class. The bar plot provides a clear visual representation of the distribution of sentiments in the dataset.
print(train_df['label'].value_counts())

import matplotlib.pyplot as plt
train_df['label'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.show()

#The accuracy scores for the Logistic Regression, Naive Bayes, and SVM models are printed to compare their performance. The accuracy score is a common metric used to evaluate classification models, representing the proportion of correctly classified instances out of the total instances. By comparing the accuracy scores of the three models, we can determine which model performs best on the test dataset for sentiment analysis.
print(f"Logistic Regression: {accuracy_score(y_test, predictions):.4f}")
print(f"Naive Bayes: {accuracy_score(y_test, nb_pred):.4f}")
print(f"SVM: {accuracy_score(y_test, svm_pred):.4f}")


#The word cloud is a visual representation of the most frequently occurring words in the positive reviews. It helps to identify common themes and sentiments expressed in the reviews. The size of each word in the word cloud corresponds to its frequency, with larger words appearing more frequently in the positive reviews. This can provide insights into what aspects of the product or service are most appreciated by customers.
from wordcloud import WordCloud

positive_sample = train_df[train_df['label'] == 1].sample(5000, random_state=42)
positive_text = " ".join(positive_sample['clean_text'])
wordcloud = WordCloud(width=800, height=400).generate(positive_text)
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive Review Word Cloud")
plt.show()

#This is the Consfusion Matrix, it is a table that is used to evaluate the performance of a classification model. It shows the number of true positives, true negatives, false positives, and false negatives, which helps to understand how well the model is performing in terms of correctly classifying positive and negative instances.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
