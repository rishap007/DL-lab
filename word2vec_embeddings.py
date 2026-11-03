"""Train a tiny Word2Vec model and inspect a learned word embedding."""

from gensim.models import Word2Vec


CORPUS = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["deep", "learning"],
]


def main() -> None:
    model = Word2Vec(
        sentences=CORPUS,
        vector_size=50,
        window=2,
        min_count=1,
        workers=1,
        seed=42,
    )

    print("Embedding for 'learning':")
    print(model.wv["learning"])

    print("\nWords similar to 'learning':")
    print(model.wv.most_similar("learning"))


if __name__ == "__main__":
    main()
