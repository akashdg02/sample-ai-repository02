import pandas as pd

print("="*50)
print("PANDAS FOR AI")
print("="*50)

data = {
    'text' : ['I love this','This is a bad product', 'Great Service', 'He is a Terrible person'],
    'sentiment' : ['positive','negative','positive','negative'],
    'score' : [0.95,0.93,0.89,0.92],
    'length' : [11,21,13,23]
}

df = pd.DataFrame(data)

# print("\n1. DataSet")
# print(df)

# print("\n2. Info")
# print(df.info())

print("\n3. Statistics ")
print(df.describe())

# print("\n4. Value counts")
# print("\nSentiment Distribution:")
# print(df['sentiment'].value_counts())

# print("\n5. Filtering")
# high_score = df[df['score'] > 0.90]
# print("\nHigh Confidence Samples:")
# print(high_score)