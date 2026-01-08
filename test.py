import requests
import random

url = "http://localhost:5000/predict"

# script.py
with open('output.txt', 'r') as file:
    feature = file.read()

feature_list = feature.splitlines()
sample_data = {}
for f in feature_list:
    sample_data[f] = random.random() # Assigning a random sample value to each feature

# print(sample_data)  
response = requests.post(url, json=sample_data)
print(response.json())
