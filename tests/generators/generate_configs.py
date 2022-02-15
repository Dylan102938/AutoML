import csv
import json

output_dir = "../"


def load_csv_to_input_structure(f_name, prediction):
    count = 3

    result = {
        "input_structure": [],
        "predictions": [],
        "joins": []
    }

    with open(f_name) as f:
        reader = csv.reader(f, delimiter=',')
        row = list(reader)[0]

        for col in row:
            if col != prediction:
                result["input_structure"].append({
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


def generate_metadata_from_kaggle_link(url):
    # ToDo
    return None
