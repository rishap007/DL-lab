"""Classify short reviews with TF-IDF features and logistic regression."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REVIEWS = [
    "I loved this movie, it was fantastic",
    "The product is excellent and very useful",
    "Amazing experience, I really enjoyed it",
    "The service was wonderful",
    "Very happy with my purchase",
    "This product is terrible",
    "I hated this movie",
    "Very disappointing experience",
    "The product broke after one day",
    "Poor quality and bad service",
]
LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
NEW_REVIEWS = [
    "The movie was amazing and I loved it",
    "The product was awful and disappointing",
    "I am very happy with this purchase",
]


def main() -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        REVIEWS,
        LABELS,
        test_size=0.2,
        random_state=42,
        stratify=LABELS,
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2%}")

    new_reviews_tfidf = vectorizer.transform(NEW_REVIEWS)
    predictions = model.predict(new_reviews_tfidf)
    print("\nNew review predictions:")
    for review, prediction in zip(NEW_REVIEWS, predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
