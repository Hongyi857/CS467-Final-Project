from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from data import preprocess_dataset


def baseline_naive_bayes(train_set, test_set):

    train_df = preprocess_dataset(train_set)
    test_df = preprocess_dataset(test_set)

    # Extract texts and labels from training and development sets
    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # Initialize TF-IDF vectorizer and transform the text data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train the Multinomial Naive Bayes classifier, default laplace smoothing = 1
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test_tfidf)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {"accuracy": accuracy, "classification_report": report}

    return nb_classifier, vectorizer, metrics
