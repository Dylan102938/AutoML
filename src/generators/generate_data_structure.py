import csv
import json

output_dir = "../../test/"
count = 3


def load_all_from_csv(f_name, prediction):
    result = {
        "inputs": [],
        "predictions": [],
        "joins": []
    }

    with open(f_name) as f:
        reader = csv.reader(f, delimiter=',')
        row = list(reader)[0]

        for col in row:
            if col != prediction:
                result["inputs"].append({
                    "name": col,
                    "format": "tabular",
                    "transformer": "none",
                    "use_input": True,
                    "location": f_name
                })
            else:
                result["predictions"].append({
                    "name": col,
                    "format": "tabular",
                    "transformer": "categorize",
                    "location": f_name
                })

    with open((output_dir + "input-" + str(count) + ".json"), "w") as f:
        json_str = json.dumps(result, indent=2)
        f.write(json_str)


if __name__ == "__main__":
    load_all_from_csv("C:/Users/dylan/Documents/AutoML/assets/credit_card/creditcard.csv", "Class")
