import requests

url = "https://raw.githubusercontent.com/raflyryhnsyh/Gemini-OCR-KTP/main/image.jpg"
response = requests.get(url)

with open("image.jpg", "wb") as f:
    f.write(response.content)