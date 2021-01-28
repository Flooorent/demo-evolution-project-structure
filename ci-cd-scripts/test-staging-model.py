from argparse import ArgumentParser
import requests
import json
import time


# maximum running time in seconds when testing staging model
timeout = 10*60

parser = ArgumentParser(description="Workspace conf")

parser.add_argument("--url", default=None, type=str, help="Workspace URL")
parser.add_argument("--pat", default=None, type=str, help="Personal Access Token")
parser.add_argument("--path", default=None, type=str, help="Absolute path to model metadata json file")

args = parser.parse_args()


# RUN NOW
worskpace_url = args.url.strip("/")
submit_url = f"{worskpace_url}/api/2.0/jobs/run-now"
headers = {"Authorization": f"Bearer {args.pat}"}

# load model version to test
with open(args.path) as json_file:
    model_metadata = json.load(json_file)

data = {
  "job_id": 40999,
  "notebook_params": {
    "model_name": model_metadata["model_name"],
    "model_version": model_metadata["model_version"],
  }
}

req = requests.post(submit_url, data=json.dumps(data), headers=headers)
run_id = req.json()["run_id"]

run_status_url = f"{worskpace_url}/api/2.0/jobs/runs/get?run_id={run_id}"
status_state = requests.get(run_status_url, headers=headers).json()["state"]

life_cycle_state = status_state["life_cycle_state"]
result_state = status_state.get("result_state")

start = time.time()
current = start
is_timeout = False

while life_cycle_state != "TERMINATED" and not is_timeout:
    time.sleep(10)
    status_state = requests.get(run_status_url, headers=headers).json()["state"]
    life_cycle_state = status_state["life_cycle_state"]
    result_state = status_state.get("result_state")
    current = time.time()
    is_timeout = True if int(current - start) >= timeout else False

if is_timeout:
    raise Exception("Testing staging model timed out.")

if not result_state:
    raise Exception("No result state, something went wrong.")

if result_state == "FAILED":
    raise Exception("Testing staging model failed")
