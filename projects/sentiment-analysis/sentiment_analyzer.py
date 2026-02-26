"""
Sentiment Analysis using Transformers
"""

from transformers import pipeline

#Loading Pretrained sentiment analysis model
classifier = pipeline('sentiment-analysis')

#Test
texts = [
    "I love Generative AI",
    "My Mentor Sucks",
    "Eh its ok"
]

for text in texts:
    result = classifier(text)[0]
    print(f"{text}")
    print(f"  ->{result['label']}: {result['score']:.2%}\n")