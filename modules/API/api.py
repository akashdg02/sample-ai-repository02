import requests
import json

print("="*60)
print("Working with Rest API's")
print("="*60)
jasonplaceholder_url="https://jsonplaceholder.typicode.com/"
req_url=jasonplaceholder_url+"posts"
print(req_url)
payload = {}
headers = {}

response = requests.request("GET", req_url, headers=headers, data=payload)

print(response.text)
posts = response.json()
print("\nTitles for User 1:")
for post in posts:
    if post['userId'] == 1:
        print(f"- {post['title']}")
