import requests
import json

print("="*60)
print("Working with Rest APIs")
print("="*60)
jsonplaceholder_url = "https://jsonplaceholder.typicode.com"
req_url = jsonplaceholder_url+"/posts/1"
print(req_url)
payload = {}
headers = {}
url = req_url
# response = requests.request("GET", url, headers=headers, data=payload)
# params={'userId':1}
# response = requests.get(f"{url}/posts",params=params)

# print(response)
# print(response.text)

print("\n3. POST Request")
new_post = {
    "userId": 1,
    "title": "Testing AI Placeholder Response",
    "body": "Body for Testing AI Placeholder Response request"
}

postresponse = requests.post(f"{url}/posts",json=new_post)

print(postresponse)
print(postresponse.text)