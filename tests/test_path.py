import requests
import os


def test_predict():
    response = requests.get(
        "http://localhost:8080/predict/", params={"path": f"{os.curdir}/tests/test_dataset.csv"})
    print(response.json())
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], list)


if __name__ == "__main__":
    test_predict()
