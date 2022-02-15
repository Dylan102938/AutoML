import json
import pandas as pd


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


def join_no_cond(p_table_1, p_table_2, col_name_1, col_name_2, join_type="inner"):
    cols_1, cols_2 = list(p_table_1.columns), list(p_table_2.columns)
    renames_1, renames_2 = {}, {}

    for col in cols_1:
        if col in cols_2:
            renames_1[col] = col + "-1"
            renames_2[col] = col + "-2"

    p_table_1 = p_table_1.rename(renames_1, axis=1)
    p_table_2 = p_table_2.rename(renames_2, axis=1)

    for col in p_table_2.columns:
        col_data = list(p_table_2[col])
        p_table_1[col] = col_data

    return p_table_1


def join_on_sorted_col(p_table_1, p_table_2, col_name_1, col_name_2, join_type="inner"):
    sorted_table_2 = p_table_2.sort_values(by=col_name_2)
    return join_no_cond(p_table_1, sorted_table_2, join_type)


def join_on_columns(p_table_1, p_table_2, col_name_1, col_name_2, join_type="inner"):
    return pd.merge(p_table_1, p_table_2, left_on=col_name_1, right_on=col_name_2, how=join_type)


def json_to_pandas(data_format):
    tables = {}
    data_format = json.loads(data_format)

    # load all data from sources and filter it
    for source in (data_format['inputs'] + data_format['predictions']):
        if source['location'] not in tables:
            if source['format'] == 'tabular': tables[source['location']] = pd.read_csv(source['location'])
            else: continue

        transformer = transformers[source['transformer']]
        table = tables[source['location']]
        col_data = table[source['name']].tolist()
        col_data = transformer(col_data)
        table[source['name']] = col_data

    # join all data as specified by join conds
    for join in (data_format['joins']):
        table1 = tables[join['source_names'][0]]
        table2 = tables[join['source_names'][1]]
        tables[join['source_names'][0]] = join_conds[join['join_cond']](table1, table2, join['join_names'][0], join['join_names'][1])
        tables.pop(join['source_names'][1])

    if len(tables) > 1:
        raise Exception("More than one table after executing joins!")

    # delete unnecessary columns
    merged_table = list(tables.values())[0]
    deleted_cols = {}
    for source in data_format['inputs']:
        if not source['use_input'] and source['name'] in merged_table.columns:
            deleted_cols[source['name']] = merged_table[source['name']].tolist()
            merged_table = merged_table.drop(source['name'], axis=1)
    for source in data_format['inputs']:
        if source['use_input'] and source['name'] not in merged_table.columns:
            merged_table[source['name']] = deleted_cols[source['name']]
            deleted_cols.pop(source['name'])

    return merged_table


transformers = {
    "none": no_transform,
    "categorize": categorize
}

join_conds = {
    "none": join_no_cond,
    "sort": join_on_sorted_col,
    "on_column": join_on_columns
}


def main():
    with open("../test/input-3.json") as f:
        data_format = f.read()

    print(json_to_pandas(data_format))


if __name__ == "__main__":
    main()
