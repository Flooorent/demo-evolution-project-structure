# Databricks notebook source
dbutils.widgets.text("input_path", "dbfs:/Users/florent.moiny@databricks.com/demo/structure-evolution/new-features-without-labels", "input_path")
dbutils.widgets.text("output_path", "dbfs:/Users/florent.moiny@databricks.com/demo/structure-evolution/predictions", "output_path")

# COMMAND ----------

from demo_evolution_project_structure.inference import make_predictions

model_name = "demo-flo-evol-struct"
input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

data_to_predict = spark.read.format("delta").load(input_path)
preds = make_predictions(data_to_predict, model_name, spark)
preds.write.format("delta").mode("overwrite").save(output_path)

# COMMAND ----------


