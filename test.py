import requests

while True:
    query = input("Query: ")
    r = requests.post("http://127.0.0.1:5000", json={"query": query})
    print(r.json()["response"])

# How can 5g technology be used for virtual reality and what other technologies are relevant?
# Tell me patents for validating hardware on site