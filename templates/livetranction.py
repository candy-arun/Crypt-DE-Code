import requests
import numpy as np
from sklearn.ensemble import IsolationForest


# API key and Ethereum address
api_key = "7YP91TPXTDTNB2ZXRS1K622A6VNMMFFM1X"
address = "0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413"

# Fetch Ethereum transaction data from Etherscan API
url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
response = requests.get(url)
data = response.json()

# Extract relevant features from the data
transactions = data['result']
features = [[float(tx['value']), float(tx['gas']), float(tx['gasPrice'])] for tx in transactions]

# Convert features to numpy array
X = np.array(features)

# Perform Isolation Forest anomaly detection
clf = IsolationForest(contamination=0.1)
outliers = clf.fit_predict(X)

# Remove outliers from the dataset
filtered_transactions = [tx for i, tx in enumerate(transactions) if outliers[i] != -1]
filtered_X = [X[i] for i, outlier in enumerate(outliers) if outlier != -1]
print("Outliers:",filtered_X)
