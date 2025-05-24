from src.eda.statistical_tests import perform_two_sample_ttest, check_normality, check_homogeneity_of_variances, perform_anova_analysis, pairwise_mannwhitneyu_tests, convert_spark_data_to_dict
from pyspark.sql.functions import mean, count, when, lit, sum as F_sum, col
from config.constants import WEEKEND_DAYS





def analyze_driver_earnings_by_gender(df):
    """
    Analyze driver earnings differences between male and female drivers.
    Business Question A analysis.
    """
    print("=" * 60)
    print("BUSINESS QUESTION A: DRIVER EARNINGS BY GENDER")
    print("=" * 60)
    
    # Filter for night time trips and calculate average per driver
    night_df = df.filter(df["IsDay"] == 0)
    
    # Approach 1: Overall average
    total_mean = night_df.agg(mean("Total_Amount")).first()[0]
    print(f"Average trip amount per night (all trips): {total_mean:.2f}")
    
    # Approach 2: Average per driver
    per_driver = night_df.groupBy("Cab_Driver_ID").agg(
        mean("Total_Amount").alias("mean_total_amount"), 
        mean("IsMale").alias("isMale")
    )
    per_driver_mean = per_driver.agg(mean("mean_total_amount")).first()[0]
    print(f"Average trip amount per driver per night: {per_driver_mean:.2f}")
    
    # Get data for statistical testing
    male_fares = df.filter(df["IsMale"] == 1).select("Total_Amount").rdd.map(lambda row: row[0]).collect()
    female_fares = df.filter(df["IsMale"] == 0).select("Total_Amount").rdd.map(lambda row: row[0]).collect()
    
    # Perform t-test
    ttest_results = perform_two_sample_ttest(
        male_fares, female_fares, 
        "Male Drivers", "Female Drivers"
    )
    
    return {
        "overall_night_average": total_mean,
        "per_driver_night_average": per_driver_mean,
        "ttest_results": ttest_results,
        "male_fares": male_fares,
        "female_fares": female_fares,
        "per_driver_df": per_driver.toPandas()
    }



def analyze_weekend_vs_weekday_earnings(df):
    """
    Analyze earnings differences between weekends and weekdays.
    Business Question B analysis.
    """
    print("=" * 60)
    print("BUSINESS QUESTION B: WEEKEND VS WEEKDAY EARNINGS")
    print("=" * 60)
    
    from config.constants import WEEKEND_DAYS
    
    # Create weekend/weekday data
    weekend_data = df.filter(df["DayOfWeek"].isin(WEEKEND_DAYS)).select("Total_Amount").rdd.map(lambda row: row[0]).collect()
    weekday_data = df.filter(~df["DayOfWeek"].isin(WEEKEND_DAYS)).select("Total_Amount").rdd.map(lambda row: row[0]).collect()
    
    print(f"Weekend trips: {len(weekend_data)}")
    print(f"Weekday trips: {len(weekday_data)}")
    
    # Perform t-test (one-tailed: weekend > weekday)
    ttest_results = perform_two_sample_ttest(
        weekend_data, weekday_data,
        "Weekend", "Weekday",
        alternative='greater'
    )
    
    return {
        "weekend_data": weekend_data,
        "weekday_data": weekday_data,
        "ttest_results": ttest_results
    }







def analyze_location_profitability(df, group_filter=None, group_name="Overall"):
    """
    Analyze profitability by pickup location.
    Business Question C analysis.
    """
    print(f"\n--- Location Analysis: {group_name} ---")
    
    # Apply group filter if provided
    if group_filter is not None:
        analysis_df = df.filter(group_filter)
    else:
        analysis_df = df
    
    # Convert to dictionary for statistical testing
    data_dict = convert_spark_data_to_dict(analysis_df, "PickUp_Colombo_ID", "Total_Amount")
    
    # Perform statistical tests
    normality_results = check_normality(data_dict)
    variance_results = check_homogeneity_of_variances(data_dict)
    anova_results = perform_anova_analysis(data_dict)
    pairwise_results = pairwise_mannwhitneyu_tests(data_dict)
    
    # Get mean earnings by location
    location_means = analysis_df.groupBy("PickUp_Colombo_ID").agg(
        mean("Total_Amount").alias("mean_earnings")
    ).orderBy("mean_earnings", ascending=False)
    
    return {
        "group_name": group_name,
        "data_dict": data_dict,
        "normality_results": normality_results,
        "variance_results": variance_results,
        "anova_results": anova_results,
        "pairwise_results": pairwise_results,
        "location_means": location_means.toPandas()
    }




def analyze_tipping_patterns(df):
    """
    Analyze tipping patterns.
    Business Question D analysis.
    """
    print("=" * 60)
    print("BUSINESS QUESTION D: TIPPING PATTERNS")
    print("=" * 60)

        
        # Calculate tipping statistics by pickup location
    tips_by_state = df.groupBy('PickUp_Colombo_ID').agg(
            (F_sum('Tip') / F_sum('Total_Amount')).alias('tip_percentage'),
            (F_sum('Tip') / count(lit(1))).alias('avg_tip'),
            (F_sum(when(col('Tip') > 0, 1).otherwise(0)) / count(lit(1))).alias("probability_of_tip"),
            count(lit(1)).alias("total_trips")
        )
    
    tips_by_state = tips_by_state.orderBy('tip_percentage', ascending=False)
    
    return tips_by_state.toPandas()




def analyze_top_routes(df, k=10):
    """
    Analyze top routes.
    Business Question E analysis.
    """
    print("=" * 80)
    print(f"TOP {k} MOST FREQUENT ROUTES ANALYSIS")
    print("=" * 80)

        # Get top k routes by frequency
    top_routes = df.groupBy('PickUp_Colombo_ID', 'DropOff_Colombo_ID').agg(
            count(lit(1)).alias('trip_count'),
            mean('Total_Amount').alias('avg_fare'),
            mean('Duration_Min').alias('avg_duration'),
            mean('Tip').alias('avg_tip')
    )
    
    top_routes = top_routes.orderBy('trip_count', ascending=False).limit(k)
    
    return top_routes.toPandas()







def analyze_passengers_vs_amount(df):
    """
    Analyze relationship between number of passengers and trip amount.
    Business Question E2 analysis.
    """
    print("=" * 80)
    print("NUMBER OF PASSENGERS VS TRIP AMOUNT ANALYSIS")
    print("=" * 80)
    
    # Calculate statistics by number of passengers
    passengers_stats = df.groupBy('N_Passengers').agg(
        mean('Total_Amount').alias('avg_amount'),
        mean('Duration_Min').alias('avg_duration'),
        mean('Tip').alias('avg_tip'),
        count(lit(1)).alias('trip_count')
    ).orderBy('N_Passengers')
    
    passengers_stats.show()
    
    # Convert to dictionary for statistical testing
    passengers_data_dict = convert_spark_data_to_dict(df, 'N_Passengers', 'Total_Amount')

    # do normality check
    normality_results = check_normality(passengers_data_dict)

    # do homogeneity of variances check
    variance_results = check_homogeneity_of_variances(passengers_data_dict)
    
    # Perform ANOVA
    anova_results = perform_anova_analysis(passengers_data_dict)

    # do pairwise tests
    pairwise_results = pairwise_mannwhitneyu_tests(passengers_data_dict)
    
    # Correlation analysis
    correlation_data = df.select('N_Passengers', 'Total_Amount').toPandas()
    correlation = correlation_data['N_Passengers'].corr(correlation_data['Total_Amount'])

    print(f'correlation: {correlation}')
    
    return passengers_stats[pairwise_results['p_value'] < 0.05]







    