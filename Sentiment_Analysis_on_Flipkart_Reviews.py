# --- Import required libraries ---
import pandas as pd                     # pandas: to load, clean, and manipulate datasets (CSV, Excel, etc.)
import nltk                             # nltk: natural language toolkit, useful for stopwords and text preprocessing
from nltk.corpus import stopwords       # stopwords: common words like "the", "and", "is" that add little meaning
from sklearn.model_selection import train_test_split  # train_test_split: divides data into training and test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # TfidfVectorizer: converts text into numerical values
from sklearn.tree import DecisionTreeClassifier      # DecisionTreeClassifier: ML algorithm for classification
from sklearn.metrics import accuracy_score, confusion_matrix # metrics to check model performance
import matplotlib.pyplot as plt         # matplotlib: to create graphs/plots
import seaborn as sns                   # seaborn: advanced plotting with prettier visuals

# --- Load dataset ---
file_path = 'flipkart_data.csv'         # name of the dataset file (must be in same folder as script)
df = pd.read_csv(file_path)             # load CSV file into a pandas DataFrame (table-like structure)

print(df.head())                        # show first 5 rows of dataset to confirm structure

# --- Download and prepare stopwords ---
nltk.download('stopwords')              # download the list of English stopwords (first time only)
stop_words = set(stopwords.words('english'))  # store stopwords in a set for fast lookup (O(1) search)

# --- Preprocess reviews: lowercase text ---
df['review'] = df['review'].str.lower() # convert all reviews to lowercase (standardize text: "Good" → "good")

# --- Function: remove stopwords from a given text ---
def remove_stopwords(text):
    if pd.isnull(text):                 # if review is missing (NaN/null), replace with empty string
        return ""
    words = text.split()                # split review into list of words (e.g. "good phone" → ["good", "phone"])
    filtered_words = []                 # create an empty list to hold words that are not stopwords
    for w in words:                     # loop through every word
        if w not in stop_words:         # keep only words that are NOT in stopwords list
            filtered_words.append(w)    # add the meaningful word into filtered list
    return " ".join(filtered_words)     # join words back into a sentence (e.g. ["good","phone"] → "good phone")

# Apply the stopword removal function to all reviews
df['review'] = df['review'].apply(remove_stopwords)

# --- Function: label sentiment based on rating ---
def label_sentiment(rating):
    # If rating is 4 or 5 → positive sentiment (1), otherwise → negative (0)
    return 1 if rating >= 4 else 0

# Apply the sentiment labeling function to the "rating" column
df['sentiment'] = df['rating'].apply(label_sentiment)

# Store cleaned dataset in a new variable for clarity
df_cleaned = df

# --- Visualize sentiment distribution ---
sentiment_counts = df_cleaned['sentiment'].value_counts()  # count how many positives and negatives
plt.figure(figsize=(6, 4))                 # set figure size
sentiment_counts.plot(kind='bar', color=['red', 'green'])  # plot bar chart: red=negative, green=positive
plt.title('Sentiment Distribution (0: Negative, 1: Positive)') # title for chart
plt.xlabel('Sentiment')                   # x-axis label
plt.ylabel('Count')                       # y-axis label
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0) # label x-axis with text
plt.show()                                # show the plot

# --- Convert reviews into numerical features using TF-IDF ---
vectorizer = TfidfVectorizer(max_features=5000)   # keep only 5000 most important words in vocabulary
X = vectorizer.fit_transform(df_cleaned['review']) # transform review text into numeric feature matrix
y = df_cleaned['sentiment']                        # target values (0 or 1, based on sentiment)

# --- Split dataset into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% of data used for training, 20% for testing
# random_state ensures results are reproducible

# --- Train Decision Tree Classifier ---
model = DecisionTreeClassifier(random_state=42)   # initialize Decision Tree
model.fit(X_train, y_train)                       # train model on training data

# --- Make predictions on test data ---
y_pred = model.predict(X_test)                    # predict sentiment for test reviews

# --- Evaluate the model ---
accuracy = accuracy_score(y_test, y_pred)          # calculate accuracy: % of correct predictions
conf_matrix = confusion_matrix(y_test, y_pred)     # confusion matrix: shows true vs predicted values

# --- Plot confusion matrix as heatmap ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues") # heatmap with numbers
plt.title('Confusion Matrix')                     # title
plt.xlabel('Predicted')                           # predicted labels
plt.ylabel('Actual')                              # actual labels
plt.show()                                        # display heatmap

print("Model Accuracy:", accuracy)                # print accuracy value in console