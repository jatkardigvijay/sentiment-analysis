from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

texts = [
    "I love this product", "This is amazing", "Best purchase ever",
    "Absolutely fantastic", "I hate this", "Terrible quality",
    "Worst experience", "Very disappointing", "It's okay",
    "Not bad, not great", "Average product"
]
labels = ["pos","pos","pos","pos","neg","neg","neg","neg","neu","neu","neu"]

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(texts, labels)

def analyze(text):
    return model.predict([text])[0]

for t in ["I really love it", "This is awful", "It's fine"]:
    print(f"{t!r} -> {analyze(t)}")