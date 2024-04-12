import json

import requests
import numpy as np

inputs = {
    "data": [
        [36, "F", 1536860201, np.nan, np.nan],
        [63, "F", 1499101932, 1504282512.0, 5180580.0],
        [49, "F", 1522429280, 1531501552.0, 9072272.0],
        [46, "U", 1507309602, 1534013038.0, 26703436.0],
        [119, "U", 1530135581, 1550261677.0, 20126096.0],
    ]
}

inputs_string = json.dumps(inputs)

inference_request = {
    "inputs": [
        {
            "name": "predict_request",
            "shape": [len(inputs_string)],
            "datatype": "BYTES",
            "data": [inputs_string],
        }
    ]
}


inference_url = "http://localhost:9000/v2/models/uplift-predictor/infer"

response = requests.post(inference_url, json=inference_request)

assert response.status_code == 200

print("OK")
