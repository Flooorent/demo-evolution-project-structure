# Databricks notebook source
# MAGIC %md ## Setup

# COMMAND ----------

from datetime import datetime

from demo_evolution_project_structure.feature_engineering import engineer_features
from demo_evolution_project_structure.inference import make_predictions
from demo_evolution_project_structure.mlflow_utils import register_best_model
from demo_evolution_project_structure.training import train

# COMMAND ----------

experiment_name = "/Users/florent.moiny@databricks.com/demo/structure-evolution/experiments/v1"
model_name = "demo-flo-evol-struct"
input_data_path = "/databricks-datasets/wine-quality/winequality-red.csv"

# COMMAND ----------

# MAGIC %md ## Read training data

# COMMAND ----------

data = spark.read.option("header", "true").option("sep", ";").csv(input_data_path)

# COMMAND ----------

# MAGIC %md ## Feature Engineering

# COMMAND ----------

data = engineer_features(data)

# COMMAND ----------

# MAGIC %md ## Training

# COMMAND ----------

now = datetime.now()
parent_run_name = now.strftime("%Y%m%d-%H%M")

train(data, experiment_name, parent_run_name)

# COMMAND ----------

# MAGIC %md ## Register best model as new production model

# COMMAND ----------

metric = "mse"
register_best_model(model_name, experiment_name, parent_run_name, metric)

# COMMAND ----------

# MAGIC %md ## Inference

# COMMAND ----------

# read new data
data_to_predict = spark.read.option("header", "true").option("sep", ";").csv(input_data_path).drop("quality")

# feature engineering
data_to_predict = engineer_features(data_to_predict)

# make predictions
preds = make_predictions(data_to_predict, model_name, spark)

# save predictions (proxy: just display preds)
display(preds)

# COMMAND ----------


