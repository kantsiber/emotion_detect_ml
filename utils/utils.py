import argparse
import json
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='config.json'
    )
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return config


def copy_json(json_input_path, json_out_path):
    with open(json_input_path, 'r', encoding='utf-8') as source_file:
        data = json.load(source_file)
    with open(json_out_path, 'w', encoding='utf-8') as dest_file:
        json.dump(data, dest_file, ensure_ascii=False, indent=2)