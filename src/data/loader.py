"""Data loading utilities for the cab analysis project."""

from pyspark.sql import SparkSession
from config.settings import SPARK_CONFIG, DATA_PATHS
import logging

def create_spark_session(app_name="CabAnalysis"):
    """
    Create and configure Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    builder = SparkSession.builder.appName(app_name)
    
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    spark = builder.master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    logging.info(f"Created Spark session: {app_name}")
    return spark

def load_cab_data(spark, file_path=None):
    """
    Load cab data from CSV file.
    
    Args:
        spark (SparkSession): Spark session
        file_path (str, optional): Path to CSV file
        
    Returns:
        DataFrame: Loaded Spark DataFrame
    """
    if file_path is None:
        file_path = DATA_PATHS["raw_data"]
    
    try:
        df = spark.read \
            .format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(file_path)
        
        record_count = df.count()
        logging.info(f"Loaded {record_count} records from {file_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

