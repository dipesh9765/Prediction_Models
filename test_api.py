import requests

# House requires 13, passing 13 fake values
print("--- HOUSE ---")
url = "http://127.0.0.1:5000/predict"
payload = {"CRIM": 0.1, "ZN": 0.1, "INDUS": 0.1, "CHAS": 0.1, "NOX": 0.1, "RM4": 0.1, "Age4": 0.1, "DIS": 0.1, "RAD": 0.1, "TAX": 0.1, "PTRATIO": 0.1, "B": 0.1, "LSTAT": 0.1}
response = requests.post(url, data=payload)
print(response.status_code)
print(response.json())

print("--- DIABETES ---")
url = "http://127.0.0.1:5000/d_predict"
payload = {"Pregnancies": 1, "Glucose": 1, "BloodPressure": 1, "SkinThickness": 1, "Insulin": 1, "BMI": 1, "DiabetesPedigreeFunction": 1, "Age": 1}
response = requests.post(url, data=payload)
print(response.status_code)
print(response.json())

