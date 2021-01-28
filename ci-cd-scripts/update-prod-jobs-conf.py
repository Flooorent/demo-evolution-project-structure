from argparse import ArgumentParser
import requests
import json


parser = ArgumentParser(
    description='Parameters used to compute most frequent pages per subdomain',
)

parser.add_argument("--url", default=None, type=str, help="Workspace URL")
parser.add_argument("--pat", default=None, type=str, help="Personal Access Token")
parser.add_argument("--dir", default=None, type=str, help="DBFS dir where build versions are stored")
parser.add_argument("--version", default=None, type=str, help="Tag version, starts with 'v'")


args = parser.parse_args()

worskpace_url = args.url.strip("/")
headers = {"Authorization": f"Bearer {args.pat}"}

inference_job_id = "41063"

build_version = args.version.lstrip("v")
build_path = f"{args.dir.strip('/')}/{args.version}/demo_evolution_project_structure-{build_version}-py3-none-any.whl"

# update python module's version
new_conf_data = {
    "job_id": inference_job_id,
    "new_settings": {
        "libraries": {
            "whl": build_path
        }
    }
}

new_job_conf_req = requests.post(f"{worskpace_url}/api/2.0/jobs/update", data=json.dumps(new_conf_data), headers=headers)
print("Updated job conf")
