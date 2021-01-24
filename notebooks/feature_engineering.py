# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

def engineer_features(df):
  return df.withColumn("total_acidity", F.col("fixed acidity") + F.col("volatile acidity"))
