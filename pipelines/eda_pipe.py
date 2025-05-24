import logging


from src.data.loader import create_spark_session, load_cab_data
from src.data.preprocessor import initial_preprocessing, create_weekend_indicator
from src.data.basic_info import display_basic_info, generate_numerical_summary, generate_categorical_summary, generate_date_summary


from src.eda.business_logic import (analyze_driver_earnings_by_gender, 
                                    analyze_weekend_vs_weekday_earnings, 
                                    analyze_location_profitability,
                                    analyze_tipping_patterns,
                                    analyze_top_routes,
                                    analyze_passengers_vs_amount)


from src.eda.visualizations import (plot_earnings_distribution_by_gender, 
                                    plot_driver_earnings_by_night, 
                                    plot_earnings_distribution_by_weekend_vs_weekday, 
                                    plot_location_earnings_comparison, 
                                    plot_tipping_heatmap)



from pyspark.sql.functions import col






class CabDataEDAWorkflow:
    """
    Comprehensive EDA workflow for cab data analysis.
    Each analysis can be run independently.
    """

    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # create spark session
        self.spark = create_spark_session()


        # load data
        self.df = load_cab_data(self.spark)

        # preprocess
        self.preprocess_data()


    def preprocess_data(self):
        self.df = initial_preprocessing(self.df)
        self.df = create_weekend_indicator(self.df)


    # part 2
    def summary_statistics(self):
        display_basic_info(self.df)

        numerical_summary = generate_numerical_summary(self.df)
        categorical_summary = generate_categorical_summary(self.df)
        date_summary = generate_date_summary(self.df)

        print("Numerical Summary:")
        print(numerical_summary)
        print("\nCategorical Summary:")
        for key, value in categorical_summary.items():
            print(f"{key}:")
            print(value)
            print("\n")
        
        print("\nDate Summary:")
        for key, value in date_summary.items():
            print(f"{key}: {value}")


    def display_data(self):
        self.df.show(5)




    # a
    def analyze_earnings_by_gender(self):
        results = analyze_driver_earnings_by_gender(self.df)
        plot_earnings_distribution_by_gender(results["male_fares"], results["female_fares"])
        plot_driver_earnings_by_night(results["per_driver_df"])

        return results


    # b
    def analyze_weekend_vs_weekday(self):
        results = analyze_weekend_vs_weekday_earnings(self.df)
        plot_earnings_distribution_by_weekend_vs_weekday(results["weekend_data"], results["weekday_data"])
        



    # c
    def analyze_location_profitability(self):
        """
        Function 4: Analyze location profitability with rankings, ANOVA, and pairwise tests.
        """
        print("=" * 80)
        print("BUSINESS QUESTION C: LOCATION PROFITABILITY ANALYSIS")
        print("=" * 80)
    
        
        # Analyze different groups
        groups = {
            'Overall': None,
            'Male Day': (col("IsMale") == 1) & (col("PickUp_Time") == "Day"),
            'Male Night': (col("IsMale") == 1) & (col("PickUp_Time") == "Night"),
            'Female Day': (col("IsMale") == 0) & (col("PickUp_Time") == "Day"),
            'Female Night': (col("IsMale") == 0) & (col("PickUp_Time") == "Night")
        }
        
        all_results = {}
        location_means_dict = {}
        
        for group_name, group_filter in groups.items():
            print(f"\n🔍 ANALYZING: {group_name.upper()}")
            print("-" * 50)
            
            # Perform analysis for this group
            group_results = analyze_location_profitability(
                self.df,
                group_filter,
                group_name
            )
            
            all_results[group_name.lower().replace(' ', '_')] = group_results
            location_means_dict[group_name] = group_results['location_means']
            
            # Generate statistical summary
            location_means_sorted = group_results['location_means'].sort_values('mean_earnings', ascending=False)
            order_by_mean = location_means_sorted['PickUp_Colombo_ID'].tolist()
            
            # Get statistically significant locations (from pairwise tests)
            pairwise_df = group_results['pairwise_results']
            if not pairwise_df.empty:
                # Get unique locations involved in significant comparisons
                significant_locations = set()
                significant_pairs = pairwise_df[pairwise_df['p_value_raw'] < 0.05]
                if not significant_pairs.empty:
                    for _, row in significant_pairs.iterrows():
                        significant_locations.add(str(row['group1']))
                        significant_locations.add(str(row['group2']))
                    
                    # Sort by significance (most involved in significant comparisons first)
                    location_significance_count = {}
                    for loc in significant_locations:
                        count = len(significant_pairs[
                            (significant_pairs['group1'] == int(loc)) | 
                            (significant_pairs['group2'] == int(loc))
                        ])
                        location_significance_count[loc] = count
                    
                    order_by_statistical_test = sorted(location_significance_count.keys(), 
                                                     key=lambda x: location_significance_count[x], 
                                                     reverse=True)
                else:
                    order_by_statistical_test = []
            else:
                order_by_statistical_test = []
            
            # Display statistical summary
            print(f"\n📊 STATISTICAL SUMMARY:")
            print(f"Order by Mean: {order_by_mean}")
            print(f"Order by statistical test: {order_by_statistical_test}")
            
        # Create comprehensive comparison visualization
        print(f"\n📈 GENERATING LOCATION COMPARISON...")
        plot_location_earnings_comparison(location_means_dict, show_plot=True)
        
        
        
        return all_results

    # d
    def analyze_tipping_patterns(self):
        """
        Function 5: Analyze tipping patterns with heatmap visualization.
        """
        tips_summary = analyze_tipping_patterns(self.df)
        
        # Display summary statistics
        print(f"\n📊 TIPPING OVERVIEW:")
        print(f"Average tip percentage across all locations: {tips_summary['tip_percentage'].mean():.1%}")
        print(f"Average tip amount: ${tips_summary['avg_tip'].mean():.2f}")
        print(f"Average probability of receiving a tip: {tips_summary['probability_of_tip'].mean():.1%}")
        
        # Create heatmap visualization
        print(f"\n📈 GENERATING TIPPING HEATMAP...")


        plot_tipping_heatmap(tips_summary, show_plot=True)
        
        return tips_summary

    
    # e.1
    def analyze_top_routes(self, k=10):
        top_routes_df = analyze_top_routes(self.df, k)


        # Display summary
        print(f"\n📊 ROUTE FREQUENCY OVERVIEW:")
        print(f"Total unique routes in dataset: {self.df.select('PickUp_Colombo_ID', 'DropOff_Colombo_ID').distinct().count()}")
        print(f"Most frequent route occurs: {top_routes_df.iloc[0]['trip_count']} times")
        print(f"Average frequency of top {k} routes: {top_routes_df['trip_count'].mean():.1f} trips")
        
        # Display top routes
        print(f"\n🏆 TOP {k} MOST FREQUENT ROUTES:")
        print("Rank | Route (Pickup → Dropoff) | Count | Avg Fare | Avg Duration | Avg Tip")
        print("-" * 75)
        
        for idx, row in top_routes_df.iterrows():
            route_name = f"{row['PickUp_Colombo_ID']} → {row['DropOff_Colombo_ID']}"
            print(f"{idx+1:4d} | {route_name:24s} | {row['trip_count']:5d} | ${row['avg_fare']:7.2f} | {row['avg_duration']:8.1f} min | ${row['avg_tip']:6.2f}")
        
        # Statistical insights
        print(f"\n📈 STATISTICAL INSIGHTS:")
        print(f"• Highest earning route: {top_routes_df.loc[top_routes_df['avg_fare'].idxmax(), 'PickUp_Colombo_ID']} → {top_routes_df.loc[top_routes_df['avg_fare'].idxmax(), 'DropOff_Colombo_ID']} (${top_routes_df['avg_fare'].max():.2f})")
        print(f"• Longest route: {top_routes_df.loc[top_routes_df['avg_duration'].idxmax(), 'PickUp_Colombo_ID']} → {top_routes_df.loc[top_routes_df['avg_duration'].idxmax(), 'DropOff_Colombo_ID']} ({top_routes_df['avg_duration'].max():.1f} min)")
        print(f"• Best tipping route: {top_routes_df.loc[top_routes_df['avg_tip'].idxmax(), 'PickUp_Colombo_ID']} → {top_routes_df.loc[top_routes_df['avg_tip'].idxmax(), 'DropOff_Colombo_ID']} (${top_routes_df['avg_tip'].max():.2f})")
        
        # Check for patterns
        popular_pickups = top_routes_df['PickUp_Colombo_ID'].value_counts()
        popular_dropoffs = top_routes_df['DropOff_Colombo_ID'].value_counts()
        
        print(f"\n🎯 LOCATION PATTERNS:")
        print(f"• Most common pickup location in top routes: Colombo ID {popular_pickups.index[0]} ({popular_pickups.iloc[0]} routes)")
        print(f"• Most common dropoff location in top routes: Colombo ID {popular_dropoffs.index[0]} ({popular_dropoffs.iloc[0]} routes)")
        
        return top_routes_df

    # e.2
    def analyze_passengers_vs_amount(self):
        """
        Function 7: Analyze relationship between number of passengers and trip amount.
        """
        
        # Perform business logic analysis
        results = analyze_passengers_vs_amount(self.df)
        return results






    def cleanup(self):
        """Clean up Spark session."""
        if self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")
            self._data_loaded = False
