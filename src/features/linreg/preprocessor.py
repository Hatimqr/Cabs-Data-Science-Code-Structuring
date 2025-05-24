from config.constants import COLUMNS_TO_DROP
from config.settings import ML_CONFIG
from pyspark.sql.functions import col, dayofweek, when, row_number
from pyspark.sql.window import Window





def get_target_column(df):
    """
    remove tip from total amount 
    """
    df = df.withColumn('Total_Amount', col('Total_Amount') - col('Tip'))
    # rename column to Fare
    df = df.withColumnRenamed('Total_Amount', 'Fare')
    return df



def add_day_of_week(df):
    """
    add day of week column
    """
    return df.withColumn("DayOfWeek", dayofweek(col("Date")))




def transform_binary_columns(df):
    """
    Transform binary columns to 0 and 1
    """
    # make IsMale and IsDay binary
    df = df.withColumn("IsMale", when(col("Gender") == "Male", 1).otherwise(0))
    df = df.withColumn("IsDay", when(col("PickUp_Time") == "Day", 1).otherwise(0))
    # remove original columns
    df = df.drop("Gender", "PickUp_Time")
    return df


def test_train_split(df):
    test_size = ML_CONFIG["test_size"]
    total_rows = df.count()
    test_rows = int(total_rows * test_size)
    train_rows = total_rows - test_rows




    # split data
    window = Window.orderBy('Date')
    df_with_row_num = df.withColumn("row_num", row_number().over(window))

    # filter train and test
    train_df = df_with_row_num.filter(col("row_num") <= train_rows)
    test_df = df_with_row_num.filter(col("row_num") > train_rows)

    # drop row_num
    train_df = train_df.drop("row_num")
    test_df = test_df.drop("row_num")


    return train_df, test_df
    # 


def remove_unnecessary_columns(df):
    """
    Remove unnecessary columns from the dataframe.
    """
    return df.drop(*COLUMNS_TO_DROP)
