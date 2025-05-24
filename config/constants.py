# Column names and types
COLUMNS = {
    "categorical": ['IsMale', 'IsDay', 'Cab_Driver_ID', 'PickUp_Colombo_ID', 'DropOff_Colombo_ID', 'DayOfWeek', 'IsWeekend'],
    "numerical": ['N_Passengers', 'Duration_Min', 'Tip', 'Total_Amount'],
    "date": ['Date']
}


# Days of week for weekend analysis
WEEKEND_DAYS = ["Saturday", "Sunday"]
WEEKDAY_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]



# ML
COLUMNS_TO_DROP = ['Cab_Driver_ID', 'Date', 'Tip']

OHE_COLUMNS = ['PickUp_Colombo_ID', 'DropOff_Colombo_ID', 'N_Passengers', 'DayOfWeek']
OTHER_COLUMNS = ['IsMale', 'IsDay', 'Duration_Min']

FEATURES = 'features'

TARGET = 'Fare'