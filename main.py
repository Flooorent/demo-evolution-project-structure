from datetime import datetime

from pyspark.sql import SparkSession

from demo_evolution_project_structure.feature_engineering import engineer_features
from demo_evolution_project_structure.inference import make_predictions
from demo_evolution_project_structure.mlflow_utils import register_best_model
from demo_evolution_project_structure.training import train


experiment_name = "/Users/florent.moiny@databricks.com/demo/structure-evolution/experiments/v1"
model_name = "demo-flo-evol-struct"
input_data_path = "/databricks-datasets/wine-quality/winequality-red.csv"

spark = SparkSession.builder.getOrCreate()

data = spark.read.option("header", "true").option("sep", ";").csv(input_data_path)
data = engineer_features(data)

now = datetime.now()
parent_run_name = now.strftime("%Y%m%d-%H%M")
train(data, experiment_name, parent_run_name)

metric = "mse"
register_best_model(model_name, experiment_name, parent_run_name, metric)

data_to_predict = spark.read.option("header", "true").option("sep", ";").csv(input_data_path).drop("quality")
data_to_predict = engineer_features(data_to_predict)
preds = make_predictions(data_to_predict, model_name, spark)

print("OK")
