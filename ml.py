import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the dataset
data = pd.read_csv('train (3).csv')

# Step 2: Feature Engineering
# Mapping for Condition/Quality (ExterQual)
qual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
data['ExterQual_Score'] = data['ExterQual'].map(qual_map).fillna(2)

# Select core features
# Adding GarageCars (from Has Garage) and PoolArea (from Has Pool)
base_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'LotArea', 'ExterQual_Score', 'GarageCars', 'PoolArea']
X = data[base_features].copy()

# One-Hot Encoding for MSZoning (Location Type)
zoning_dummies = pd.get_dummies(data['MSZoning'], prefix='Zone')
X = pd.concat([X, zoning_dummies], axis=1)

target = data['SalePrice']

# Save the exact column order for backend compatibility
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save the trained model
joblib.dump(model, 'house_price_model.pkl')

print("Advanced Model Training Complete (V2).")
print(f"Features used: {model_columns}")
