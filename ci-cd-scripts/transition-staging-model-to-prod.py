from argparse import ArgumentParser
import requests
import json


parser = ArgumentParser(
    description='Parameters used to compute most frequent pages per subdomain',
)

parser.add_argument("--url", default=None, type=str, help="Workspace URL")
parser.add_argument("--pat", default=None, type=str, help="Personal Access Token")
parser.add_argument("--path", default=None, type=str, help="Absolute path to model metadata json file")

args = parser.parse_args()

worskpace_url = args.url.strip("/")

headers = {"Authorization": f"Bearer {args.pat}"}


# load model version to test
with open(args.path) as json_file:
    model_metadata = json.load(json_file)


# retrieve current prod model version
data = {
    "name": model_metadata["model_name"],
    "stages": ["Production"]
}

old_prod_model = requests.get(f"{worskpace_url}/api/2.0/mlflow/registered-models/get-latest-versions", data=json.dumps(data), headers=headers)
old_prod_model_version = old_prod_model.json()["model_versions"][0]["version"]
print(f"Retrieved current model version in production: {old_prod_model_version}")


# transition staging model to prod
new_model_to_prod_data = {
    "name": model_metadata["model_name"],
    "version": model_metadata["model_version"],
    "stage": "Production",
    "archive_existing_versions": "False",
}

new_model_to_prod_req = requests.post(f"{worskpace_url}/api/2.0/mlflow/model-versions/transition-stage", data=json.dumps(new_model_to_prod_data), headers=headers)
print("Moved staging model to production")


# transition old prod model to archived
old_prod_model_to_archived_data = {
    "name": model_metadata["model_name"],
    "version": old_prod_model_version,
    "stage": "Archived",
    "archive_existing_versions": "False",
}

old_prod_model_to_archived_req = requests.post(f"{worskpace_url}/api/2.0/mlflow/model-versions/transition-stage", data=json.dumps(old_prod_model_to_archived_data), headers=headers)
print("Moved old production model to archived")
