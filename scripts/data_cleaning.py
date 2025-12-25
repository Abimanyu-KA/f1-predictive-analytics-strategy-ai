import pandas as pd

# Define base paths
raw_path = r'data\raw\\'
processed_path = r'data\processed\\'

# Helper function to read, clean, and save CSV files
def clean_and_save(filename, columns_to_remove=None, na_values=['\\N', '']):
    df = pd.read_csv(raw_path + filename, na_values=na_values)
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
    df.to_csv(processed_path + filename, index=False)

# Process files
clean_and_save('circuits.csv', ['url'])
clean_and_save('constructor_results.csv', ['constructorResultsId'])
clean_and_save('constructor_standings.csv', ['constructorStandingsId', 'positionText'])
clean_and_save('constructors.csv', ['url'])
clean_and_save('driver_standings.csv', ['driverStandingsId', 'positionText'])
clean_and_save('drivers.csv', ['url'])
clean_and_save('lap_times.csv')  # No columns to remove
clean_and_save('pit_stops.csv', ['time', 'duration'])
clean_and_save('qualifying.csv', ['qualifyId', 'number'])
clean_and_save('races.csv', [
    'url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time',
    'quali_date', 'quali_time', 'sprint_date', 'sprint_time'
])
clean_and_save('results.csv', ['resultId', 'number', 'time'])
clean_and_save('sprint_results.csv', ['resultId', 'number', 'time'])
clean_and_save('status.csv')  # No columns to remove
