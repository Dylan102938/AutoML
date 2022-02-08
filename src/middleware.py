import json
import pandas as pd
from helper import *


def no_transform(x):
    return x


def categorize(x):
    options = {}
    curr_num = 0
    for i, entry in enumerate(x):
        if entry not in options:
            options[entry] = curr_num
            curr_num += 1

        x[i] = options[entry]

    return x


def json_to_pandas(data_format):
    tables = {}
    data_format = json.loads(data_format)

    for source in (data_format['inputs'] + data_format['predictions']):
        if source['location'] not in tables:
            if source['format'] == 'tabular': tables[source['location']] = pd.read_csv(source['location'])
            else: continue

        transformer = transformers[source['transformer']]
        table = tables[source['location']]
        col_data = table[source['name']].tolist()
        col_data = transformer(col_data)
        table[source['name']] = col_data

    for table_name in tables:
        print(tables[table_name].head())


transformers = {
    "none": no_transform,
    "categorize": categorize
}


def main():
    with open("../test/input-2.json") as f:
        data_format = f.read()

    json_to_pandas(data_format)


if __name__ == "__main__":
    main()
