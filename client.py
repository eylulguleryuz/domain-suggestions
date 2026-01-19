import requests
import json

# ------------------ #
#   REQUEST EXAMPLE
# ------------------ #

url = "http://127.0.0.1:8000/generate"  # make sure FastAPI is running
payload = {
    "userinput": "We are the lighthouse in a sea of data, guiding your brand toward the shores of untapped potential."
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Response JSON:")
    # Pretty-print JSON with indentation
    print(json.dumps(data, indent=4, ensure_ascii=False))
else:
    print("Error:", response.status_code, response.text)
