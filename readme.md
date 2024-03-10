## DE-Anonymisation for monitoring and tracking with cryptocurrencies transaction technolgy

## Import Packages
 requests
 numpy
 sklearn.ensemble
 sklearn.cluster
  matplotlib.pyplot


## Steps to follow:
  step1: run app.py [python app.py]
  step2: copy the url from terminal [http://127.0.0.1:5000]
  step3: login page with username and password [test,test]
  step4: click on live transactions button
  step5: it directs to a page where live ETH users Value , Gas , GasPrices will be displayed in a list of arrays
  step6: transaction history button directs to the graph of user transaction .
  step7: Click on clusters it directs to see Number of transactions,Mean transaction amount,Mean gas price.,Mean gas used in the terminal
  step8: Comparing monitored data and user data clusters can be identified into individual user 

  At this stage we have DE-ANONYMISED and we have found the user with illegal transactions(Just a prototype)


  ## Description 

  unusual activity in the transaction which is detected using scikit-learn .
  Outliars being mentioned here as -1 and they are represented as unusual activities.
  Ethereum transfer datas are collected using etherscan Tool and request library.
