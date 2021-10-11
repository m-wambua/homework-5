
import requests

url = 'http://127.0.0.1:5000/predict1'
customer_id ='xyz-123'
customer = {

"contract": "two_year",  
"tenure": 1, 
"monthlycharges": 10
}

response =requests.post(url,json=customer).json()
print(response)
if response['churn']==True:
    print('sending promo email to %s' % customer_id)
    
else:
    print('do not send the promotion email to %s' % customer_id)




