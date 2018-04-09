from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

train = pd.read_csv('data/trainReviews.tsv', header=0, delimiter="\t", quoting=3)

#vectorizer = CountVectorizer()

vectorizer = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=10, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)

train_data_features = vectorizer.fit_transform(train["text"])

#clf = RandomForestClassifier()
#clf = MLPClassifier()
#clf = MultinomialNB()
#clf = LogisticRegression()
#clf = SVC()

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

clf.fit( train_data_features, train["category"] )

test = pd.read_csv('data/testReviews.tsv', header=0, delimiter="\t", quoting=3)

test_data_features = vectorizer.transform(test['text'])

binary_predictions = clf.predict(test_data_features)

prediction_list = pd.DataFrame( data={"id":test["id"], "category":test['category'], "prediction":binary_predictions} )

print(1 - (sum(np.absolute(prediction_list['category'] - prediction_list['prediction'])) / 500))


