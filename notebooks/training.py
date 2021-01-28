# Databricks notebook source
dbutils.widgets.text("input_path", "dbfs:/Users/florent.moiny@databricks.com/demo/structure-evolution/features", "input_path")

# COMMAND ----------

NB_VERSION = "0.9.0"


from datetime import datetime
from demo_evolution_project_structure.mlflow_utils import register_best_model
from demo_evolution_project_structure.training import train

experiment_name = "/Users/florent.moiny@databricks.com/demo/structure-evolution/experiments/v1"
model_name = "demo-flo-evol-struct"
input_path = dbutils.widgets.get("input_path")

# COMMAND ----------

# MAGIC %md ## Train new model

# COMMAND ----------

data = spark.read.format("delta").load(input_path)

now = datetime.now()
parent_run_name = now.strftime("%Y%m%d-%H%M")

train(data, experiment_name, parent_run_name)

# COMMAND ----------

# MAGIC %md ## Register new model version

# COMMAND ----------

metric = "mse"
model_details = register_best_model(model_name, experiment_name, parent_run_name, metric)

# COMMAND ----------

print(f"New version: {model_details.version}")
