"""Data preprocessing utilities for the cab analysis project."""

from pyspark.sql.functions import when, col, date_format
import logging

def initial_preprocessing(df):
    """
    Perform initial data preprocessing.
    
    Args:
        df (DataFrame): Raw Spark DataFrame
        
    Returns:
        DataFrame: Preprocessed Spark DataFrame
    """
    logging.info("Starting initial preprocessing...")
    
    # Convert gender to binary variable
    df = df.withColumn("IsMale", when(df["Gender"] == "M", 1).otherwise(0))
    df = df.drop("Gender")

    # convert day time to binary variable
    df = df.withColumn("IsDay", when(df["PickUp_Time"] == "Day", 1).otherwise(0))
    df = df.drop("PickUp_Time")
    
    # Convert pickup and dropoff IDs to string (for later one-hot encoding)
    df = df.withColumn("PickUp_Colombo_ID", df["PickUp_Colombo_ID"].cast("string"))
    df = df.withColumn("DropOff_Colombo_ID", df["DropOff_Colombo_ID"].cast("string"))
    
    # Add day of week column
    df = df.withColumn("DayOfWeek", date_format(col("Date"), "EEEE"))
    
    logging.info("Initial preprocessing completed")
    return df

def create_weekend_indicator(df):
    """
    Add weekend/weekday indicator to the DataFrame.
    
    Args:
        df (DataFrame): Spark DataFrame
        
    Returns:
        DataFrame: DataFrame with weekend indicator
    """
    from config.constants import WEEKEND_DAYS
    
    df = df.withColumn("IsWeekend", 
                      when(df["DayOfWeek"].isin(WEEKEND_DAYS), 1).otherwise(0))
    
    return df

