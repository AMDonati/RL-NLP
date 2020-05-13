import json
import argparse


def extract_short_json(json_data_path, out_path, num_questions):
    with open(json_data_path, 'r') as f:
        questions = json.load(f)['questions']
    select_questions = questions[:num_questions]
    out_json = {'questions': select_questions}

    with open(out_path, 'w') as f:
        json.dump(out_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, required=True, help="path for CLEVR questions json files")
    parser.add_argument("-json_out_path", type=str, required=True, help="path out json file with reduced dataset")
    parser.add_argument('-num_samples', type=int, required=True,
                        help="used to select a subset of the whole CLEVR dataset")

    args = parser.parse_args()

    extract_short_json(out_path=args.json_out_path, json_data_path=args.data_path, num_questions=args.num_samples)
