from scikit.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

print("="*60)
print("SCIKIT LEARN CLASSIFICATION")
print("="*60)

np.random.seed(42)

#Feature [hours studented, previous score]

x = np.array([
    [2,45],[3,55],[5,65],[7,75],[8,85],[1,40],[4,60],[6,70],[9,90],[10,95],[2.5,50],[5.5,68],[7.5,78],[8.5,88],[9.5,92]
])

#Labels: 0 = Fail, 1 = Pass

y = np.array([0,0,1,1,1,0,1,1,1,1,0,1,1,1,1])

print(f"\n1. DATASET")
print(f"Samples: {len(x)}")
print(f"Features : {x.shpae[1]}")
print(f"First 5 samples:")
print(x[:5])