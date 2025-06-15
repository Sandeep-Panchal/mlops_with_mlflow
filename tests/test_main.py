import sys
sys.path.append("../")

import json

from src.app import app

client = app.test_client()

def test_predict_houseage():
    response = client.post(
        "/predict",
        data=json.dumps({"HouseAge": [25]}),
        content_type="application/json"
    )
    
    assert response.status_code == 200
    assert "prediction" in response.get_json()