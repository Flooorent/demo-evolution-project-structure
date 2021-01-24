# Databricks notebook source
# MAGIC %md ## Setup

# COMMAND ----------

import hyperopt as hp
from hyperopt import fmin, rand, tpe, hp, SparkTrials, STATUS_OK
import mlflow.sklearn
from  mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time

# COMMAND ----------

experiment_name = "/Users/florent.moiny@databricks.com/demo/structure-evolution/experiments/v1"
mlflow.set_experiment(experiment_name)

model_name = "demo-flo-evol-struct"

input_data_path = "/databricks-datasets/wine-quality/winequality-red.csv"

# COMMAND ----------

# MAGIC %md ## Read training data

# COMMAND ----------

data = spark.read.option("header", "true").option("sep", ";").csv(input_data_path)

# COMMAND ----------

# MAGIC %md ## Feature Engineering

# COMMAND ----------

data = data.withColumn("total_acidity", F.col("fixed acidity") + F.col("volatile acidity"))

# COMMAND ----------

# MAGIC %md ## Training

# COMMAND ----------

data = data.toPandas()
X_train, X_test, y_train, y_test = train_test_split(data.drop(["quality"], axis=1), data[["quality"]].values.ravel(), random_state=42)

# COMMAND ----------

def evaluate_hyperparams(params):
  min_samples_leaf = int(params['min_samples_leaf'])
  max_depth = params['max_depth']
  n_estimators = int(params['n_estimators'])
  
  rf = RandomForestRegressor(
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    n_estimators=n_estimators,
  )
  rf.fit(X_train, y_train)
  
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  predictions = rf.predict(X_test)
  mse = mean_squared_error(y_test, predictions)
  mae = mean_absolute_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)
  
  mlflow.log_metric("mse", mse)
  mlflow.log_metric("mae", mae)  
  mlflow.log_metric("r2", r2)

  return {'loss': mse, 'status': STATUS_OK}


search_space = {
  'n_estimators': hp.uniform('n_estimators', 10, 100),
  'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 20),
  'max_depth': hp.uniform('max_depth', 2, 10),
}

spark_trials = SparkTrials(parallelism=4)
run_name = "v1"

with mlflow.start_run(run_name=run_name):
  argmin = fmin(
    fn=evaluate_hyperparams,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md ## Register best model as new production model

# COMMAND ----------

client = MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

parent_run = client.search_runs(experiment_id, filter_string=f"tags.mlflow.runName = '{run_name}'", order_by=["metrics.loss ASC"])[0]
parent_run_id = parent_run.info.run_id
best_run_from_parent_run = client.search_runs(experiment_id, filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'", order_by=["metrics.mse ASC"])[0]

best_model_uri = f"runs:/{best_run_from_parent_run.info.run_id}/random-forest-model"
model_details = mlflow.register_model(model_uri=best_model_uri, name=model_name)

time.sleep(5)

client.transition_model_version_stage(model_details.name, model_details.version, stage="production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md ## Inference

# COMMAND ----------

# read new data
data_to_predict = spark.read.option("header", "true").option("sep", ";").csv(input_data_path).drop("quality")

# feature engineering
data_to_predict = data_to_predict.withColumn("total_acidity", F.col("fixed acidity") + F.col("volatile acidity"))

# read production model
model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/Production")
model_version = client.get_latest_versions(model_name, stages=["Production"])[0].version

# make predictions
preds = (
  data_to_predict
    .withColumn("prediction", model_udf(*data_to_predict.columns))
    .withColumn("model_version", F.lit(model_version))
)

# save predictions (proxy: just display preds)
display(preds)

# COMMAND ----------


