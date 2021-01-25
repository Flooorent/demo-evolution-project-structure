# Databricks notebook source
dbutils.widgets.text("input_path", "/databricks-datasets/wine-quality/winequality-red.csv", "input_path")
dbutils.widgets.text("output_path", "dbfs:/Users/florent.moiny@databricks.com/demo/structure-evolution/features", "output_path")

# COMMAND ----------

from demo_evolution_project_structure.feature_engineering import engineer_features
from pyspark.sql import functions as F


input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

data = spark.read.option("header", "true").option("sep", ";").csv(input_path)
features = engineer_features(data)
renamed_columns = [F.col(col_name).alias(col_name.replace(" ", "_")) for col_name in features.columns]
features.select(renamed_columns).write.format("delta").mode("overwrite").save(output_path)

# COMMAND ----------


