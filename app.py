from flask import Flask, render_template
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

@app.route('/livetranction.html')
def livetranction():
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

     return render_template("livetranction.html",fx=filtered_X)



@app.route('/graph.html')
def graph():
    api_key = "7YP91TPXTDTNB2ZXRS1K622A6VNMMFFM1X"
    address = "0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413"
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

    max_transactions = 1000  # Limit the number of transactions to process

# Assume 'timestamps' and 'amounts' contain transaction data
    timestamps = [int(tx['timeStamp']) for tx in transactions[:max_transactions]]
    amounts = [float(tx['value']) for tx in transactions[:max_transactions]]

    plt.plot(timestamps, amounts)
    plt.xlabel('Timestamp')
    plt.ylabel('Amount (ETH)')
    plt.title('Ethereum Transaction History')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    graph=plt.show()


    return render_template('graph.html',graph=graph)


@app.route('/clusters.html')
def cluster():
    api_key = "7YP91TPXTDTNB2ZXRS1K622A6VNMMFFM1X"
    address = "0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413"
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
    
    
    num_clusters = 5  # Number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(filtered_X)
    cluster_labels = kmeans.labels_

# Analyze characteristics of each cluster
    cluster_data = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_data:
           cluster_data[label] = []
        cluster_data[label].append(filtered_transactions[i])


    threshold=10e+20
# Now, compare the characteristics of clusters with external data sources
# Replace the external_data dictionary with actual data
    external_data = {
         "user1": {"mean_transaction_amount": 1000},
         "user2": {"mean_transaction_amount": 5000},
         "user3": {"mean_transaction_amount": 1000},
         "user4": {"mean_transaction_amount": 7500},
         "user5": {"mean_transaction_amount": 26000},
         "user6": {"mean_transaction_amount": 1900},
    # Add more users and their information as needed
}

# Compare characteristics of each cluster with external data
    for cluster, transactions in cluster_data.items():
       cluster1=print(f"Cluster {cluster + 1}:")
       nocluster=print(f"Number of transactions: {len(transactions)}")
       mean=print(f"Mean transaction amount: {np.mean([float(tx['value']) for tx in transactions])}")
       meangasprice=print(f"Mean gas price: {np.mean([float(tx['gasPrice']) for tx in transactions])}")
       meangas=print(f"Mean gas used: {np.mean([float(tx['gas']) for tx in transactions])}")

    # Compare cluster characteristics with external data
    for user, user_data in external_data.items():
        # Example: Compare mean transaction amount
        if abs(np.mean([float(tx['value']) for tx in transactions]) - user_data['mean_transaction_amount']) < threshold:
            print(f"Cluster {cluster + 1} matches user {user}")


    return render_template('clusters.html',cluster1=cluster1,nocluster=nocluster,mean=mean,meangasprice=meangasprice,meangas=meangas,user=user)




if __name__ == '__main__':
    app.run(debug=True)
