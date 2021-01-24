# demo-evolution-project-structure

Demo: project structure evolution from notebooks to python module

## Running tests

To run tests:
```
pytest
```

## Creating the distribution

To create the distribution:
```
python setup.py sdist bdist_wheel
```

## To push the distribution and the main script to DBFS
Install the `databricks-cli` and set up the workspace url and the personal access token (via a profile for instance). Then run:
```
databricks fs cp /path/to/version/demo_evolution_project_structure-0.0.3-py3-none-any.whl dbfs:/path/in/dbfs/versions/0.0.3/demo_evolution_project_structure-0.0.3-py3-none-any.whl --profile <your_profile>
databricks fs cp /path/to/main/file/main.py dbfs:/path/in/dbfs/versions/0.0.3/main.py --profile <your_profile>
```
