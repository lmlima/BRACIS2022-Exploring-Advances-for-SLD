import re
import pandas as pd
from pathlib import Path
import argparse


def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    # set up regular expressions
    # use https://regexper.com to visualise these if required
    rx_dict = {
        'loss': re.compile(r'- Loss: (?P<loss>.*)\n'),

        'accuracy': re.compile(r'- Accuracy: (?P<accuracy>.*)\n'),
        'top2accuracy': re.compile(r'- Top 2 accuracy: (?P<top2accuracy>.*)\n'),
        'baccuracy': re.compile(r'- Balanced accuracy: (?P<baccuracy>.*)\n'),

        'confusion_matrix': re.compile(r'- Confusion Matrix:(?P<confusion_matrix>.*)\n'),

        'weighted_avg': re.compile(
            r'weighted avg       (?P<precision>\d+\.\d+)      (?P<recall>\d+\.\d+)      (?P<f1score>\d+\.\d+) .*\n'),

        'auc': re.compile(r'- AUC: (?P<auc>\d+\.\d+)'),

    }

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None


def parse_file(filepath):
    with open(filepath, 'r') as file_object:
        lines = file_object.readlines()

        data = {}
        for line in lines:
            # print(line)
            key, match = _parse_line(line)

            if key in ['loss', 'accuracy', 'top2accuracy', 'baccuracy', "auc", "weighted_avg"]:
                group_dict = match.groupdict()
                group_dict.update((k, float(v)) for k, v in group_dict.items())
                data.update(group_dict)

    return data


def metrics_dataframe(path='/tmp/results', output_path='/tmp'):
    file_list = list(Path(path).glob('*/best_metrics/metrics.txt'))

    data_list = []

    for filepath in file_list:
        data_file = parse_file(filepath)

        regex = r'(?P<fusion>[A-Za-z0-9-]+)_(?P<model>[A-Za-z0-9-_]+)_reducer_(?P<reducer>\d+)_fold_(?P<fold>\d+)_\d+'
        match = re.compile(regex).search(filepath.parent.parent.name)

        data_file.update(match.groupdict())
        data_list.append(data_file)

    data = pd.DataFrame(data_list)
    data.set_index(['model', 'fusion', 'reducer', 'fold'], inplace=True)

    data_desc = data.groupby(["model", "fusion", "reducer"]).describe()
    data_desc.to_csv(Path(output_path, "result_desc.csv"))

    simplified_data = data_desc.drop(['max', 'min', '25%', '50%', '75%', 'count'], axis=1, level=1)
    data_desc.to_csv(Path(output_path, "result_desc_small.csv"))
    simplified_data.to_string(Path(output_path, "result_desc_small.txt"))

    print(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('result_path', default='/tmp/results', nargs='?',
                        help='Path to result directories')
    parser.add_argument('output_path', default='/tmp', nargs='?',
                        help='Path to output results summary ')

    args = parser.parse_args()

    metrics_dataframe(args.result_path, args.output_path)
