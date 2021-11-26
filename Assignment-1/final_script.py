import pandas as pd
import numpy as np
import re
import string
import nltk
import unicodedata
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

csv_name = "final_categories_csv.csv"
df = pd.read_csv(csv_name)
df.dropna(inplace=True)
print(df["label"].value_counts())
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    html = re.compile(r"<.*?>")
    table = str.maketrans("", "", string.punctuation)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    # t1 = url.sub(r"",text)
    t2 = html.sub(r"", text)
    t3 = emoji_pattern.sub(r"", t2)
    t4 = t3.translate(table)
    res = ''.join([i for i in t4 if not i.isdigit()])
    return res


## removing all non-letter values to avoid error of getting "expected string or bytes-like object "

def zumbaa(text):
    letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",  # Replace all non-letters with spaces
                          str(text))
    return letters_only


CLEANR = re.compile('<.*?>')


def basic_clean(text):
    """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


df["body"] = df["body"].apply(lambda text: clean_text(text))
df["body"] = df["body"].apply(lambda text: zumbaa(text))
df["body"] = df["body"].apply(lambda text: ' '.join(text.split()))
df["body"] = df["body"].apply(lambda text: basic_clean(text))
df["body"] = df["body"].apply(lambda text: ' '.join(text))

x1 = df['body'] + df['sender'] + df['subject']
y1 = df['label']

X_train, X_val, y_train, y_val = train_test_split(x1, y1, test_size=0.25)

### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             stop_words='english', ngram_range=(2, 3))
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_val)
tfdif_tokens = vectorizer.get_feature_names()
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X_train_transformed, y_train)
X_train_transformed = selector.transform(X_train_transformed).toarray()
X_test_transformed = selector.transform(X_test_transformed).toarray()
feature_names = [tfdif_tokens[i] for i in selector.get_support(indices=True)]
print("feature_names:", feature_names)
# Decision Trees
clf = tree.DecisionTreeClassifier(min_samples_split=20, max_depth=40)
clf.fit(X_train_transformed, y_train)
pred = clf.predict(X_test_transformed)
print("Decision Accuracy Score", accuracy_score(y_val, pred))
# SVC
from sklearn.svm import SVC

clf = SVC(kernel="poly")
clf.fit(X_train_transformed, y_train)
pred = clf.predict(X_test_transformed)
print("SVC Accuracy Score", accuracy_score(y_val, pred))
# Gaussian NBC
clf = GaussianNB()
clf.fit(X_train_transformed, y_train)
pred = clf.predict(X_test_transformed)
print("GNB Accuracy Score", accuracy_score(y_val, pred))
# RFC
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=50, random_state=42)
clf.fit(X_train_transformed, y_train)
pred = clf.predict(X_test_transformed)
print("RFC Accuracy Score", accuracy_score(y_val, pred))