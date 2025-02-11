import requests

url = "http://gdgoc-demo.vercel.app/ask"
data = {
    "question": "What is the capital of France?"
}

response = requests.post(url, json=data)

# Print the response content
print(response.content)

# Check if the response is JSON
try:
    response_json = response.json()
    print(response_json)
except ValueError:
    print("Response is not valid JSON")