# Databricks notebook source
# MAGIC %md # In Staging: make predictions on test data with staging model
# MAGIC 
# MAGIC This notebook:
# MAGIC - retrieves Staging model
# MAGIC - loads test data
# MAGIC - make predictions with Staging model on test data
# MAGIC 
# MAGIC NB/TODO: theoretically we should make sure that it:
# MAGIC - doesn't break anything
# MAGIC - is better than the model in production

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

NB_VERSION = "0.9.0"

dbutils.widgets.text("model_name", "", "model_name")
dbutils.widgets.text("model_version", "", "model_version")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import mean_squared_error

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

input_data_path = "dbfs:/Users/florent.moiny@databricks.com/demo/structure-evolution/features"

# COMMAND ----------

# MAGIC %md ## Read new model version

# COMMAND ----------

client = MlflowClient()

model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %md ## Model's stage should be None or Staging, if not then we're using the wrong model version

# COMMAND ----------

current_stage = client.get_model_version(model_name, model_version).current_stage
if current_stage in ["Production", "Archived"]:
  raise Exception(f"Bad current stage '{current_stage}' for model version {model_version}. Should be None or Staging.")

# COMMAND ----------

# MAGIC %md ## Make predictions on test data

# COMMAND ----------

data = spark.read.format("delta").load(input_data_path)
preds = data.withColumn("prediction", model_udf(*data.drop("quality").columns)).select("quality", "prediction").toPandas() # it's okay since dataframe is small
mse = mean_squared_error(preds["quality"], preds["prediction"])
print(f"MSE: {mse}")

# COMMAND ----------

# MAGIC %md ## Test succeeded: transition the model to Staging

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage="Staging",
)

# COMMAND ----------


