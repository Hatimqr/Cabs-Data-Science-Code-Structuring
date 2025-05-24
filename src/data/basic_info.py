from pyspark.sql.functions import col, count, countDistinct, min, max
from config.constants import COLUMNS




def display_basic_info(df):
    """
    Display basic information about the dataset.
    
    Args:
        df (DataFrame): Spark DataFrame
    """
    print("=== Dataset Basic Information ===")
    print(f"Number of records: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nSchema:")
    df.printSchema()
    print("\nFirst 5 rows:")
    df.show(5)



def generate_numerical_summary(df):
    """
    Generate summary statistics for numerical columns.
    
    Args:
        df (DataFrame): Spark DataFrame
        
    Returns:
        DataFrame: Summary statistics as Pandas DataFrame
    """
    
    numerical_df = df.select(COLUMNS["numerical"])
    
    # # Convert string IDs to integers for summary
    # for col_name in numerical_df.columns:
    #     numerical_df = numerical_df.withColumn(col_name, numerical_df[col_name].cast("int"))
    
    # Generate summary statistics
    summary = numerical_df.summary()
    
    return summary.toPandas()

def generate_categorical_summary(df):
    """
    Generate summary for categorical columns.
    
    Args:
        df (DataFrame): Spark DataFrame
        
    Returns:
        dict: Dictionary containing count distributions for categorical columns
    """
    categorical_summaries = {}
    
    # Define categorical columns to analyze
    categorical_cols = COLUMNS["categorical"]
    
    for col_name in categorical_cols:
        if col_name in df.columns:
            # Get value counts for each categorical column
            counts = df.groupBy(col_name).count().orderBy("count", ascending=False)
            categorical_summaries[col_name] = counts.toPandas()
    
    return categorical_summaries

def generate_date_summary(df):
    """
    Generate summary for date columns.
    
    Args:
        df (DataFrame): Spark DataFrame
        
    Returns:
        dict: Date range and distribution summary
    """
    date_summary = df.select(
        min(col("Date")).alias("earliest_date"),
        max(col("Date")).alias("latest_date"),
        countDistinct(col("Date")).alias("distinct_dates_count"),
        count(col("Date")).alias("total_dates_count")
    ).first()
    
    return {
        "earliest_date": date_summary["earliest_date"],
        "latest_date": date_summary["latest_date"],
        "distinct_dates": date_summary["distinct_dates_count"],
        "total_records": date_summary["total_dates_count"]
    }