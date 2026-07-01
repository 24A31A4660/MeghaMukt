import requests
import json

import os

# Path to the image relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', '..', 'assets', 'test-cloudy-satellite.png')

url = "http://localhost:8000/api/process-satellite-image"
files = {'image': open(image_path, 'rb')}
print("Sending POST upload request to backend server...")
r = requests.post(url, files=files)

print("Response status code:", r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Failed to parse JSON response:", e)
    print(r.text)
