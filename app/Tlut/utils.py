import yaml
import json

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def store_json(json_path, data, indent=None):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)