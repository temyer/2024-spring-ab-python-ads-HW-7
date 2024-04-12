import json
import sys
import os


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["solo", "two"]:
        print("The script takes exactly 1 argument: {solo, two}")
        exit(-1)

    filename = "model-settings.json"

    data = {
        "name": "uplift-predictor",
        "implementation": "inference.UpliftPredictor",
        "parameters": {"uri": ""},
    }

    if sys.argv[1] == "solo":
        data["parameters"]["uri"] = "solo_cb.joblib"
    else:
        data["parameters"]["uri"] = "two_cb.joblib"

    os.remove(filename)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


main()
