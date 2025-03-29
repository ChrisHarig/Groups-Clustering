# Transforms the csv's in the best_times directory in to a single grouped csv
# csv's must be in the names FirstLast.csv, with only an Event column and Time column

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Creates a table of all swimmers times in all events
def make_times_df(save_csv: bool):
    times_df = pd.DataFrame() # grouped dataframe

    best_times_folder = Path("./best_times") # grabs the current data  

    # Add each swimmer to the table
    for file_path in best_times_folder.glob('*.csv'):
        try:
            df = pd.read_csv(file_path)
        except Exception as e: 
            print(f"Error processing {file_path}")
        
        name = str(file_path).split('.csv')[0].replace('best_times/', '') # grab the name of the swimmer
        
        row_df = pd.DataFrame([[name]], columns=['Name'])

        for index, row in df.iterrows():
            event = row['Event']
            time = row['Time']
            row_df[event] = time

        times_df = pd.concat([times_df, row_df], ignore_index=True) # concat with the grouped dataframe

    if save_csv == True:
        times_df.to_csv("times_data.csv", index=False)

    return times_df

# Remove non-yards entries and non-NCAA events from the times data
def make_yards_times_df(times_df: pd.DataFrame, save_csv: bool):
    name_column = times_df.columns[0]
    yard_columns = [col for col in times_df.columns if 'Y' in col]
    filtered_yards_columns = [col for col in yard_columns if not any(num in col for num in ['150', '300', '600', '50 Y Fly', '50 Y Back', '50 Y Breast', '100 Y IM', '1M','3M'])]
    new_columns = [name_column] + filtered_yards_columns

    yards_times_df = times_df[new_columns]

    if save_csv == True:
        yards_times_df.to_csv("yards_time_data.csv", index=False)

    return yards_times_df

# Create a table of all times as a percent of the NCAA record
def normalize_to_NCAA_record(yards_times_df: pd.DataFrame, save_csv: bool):
    # NCAA records in seconds
    NCAA_records = {
        '50 Y Free': 17.63,
        '100 Y Free': 39.90,
        '200 Y Free': 88.81,
        '500 Y Free': 242.31,
        '1000 Y Free': 513.93,
        '1650 Y Free': 852.08,
        '100 Y Back': 43.35,
        '200 Y Back': 95.57,
        '100 Y Breast': 49.53,
        '200 Y Breast': 106.35,
        '100 Y Fly': 42.80,
        '200 Y Fly': 97.35,
        '200 Y IM': 96.34,
        '400 Y IM': 208.82
    }

    reformatted_yards_times_df = yards_times_df.copy()
    # Convert times to 
    for col in reformatted_yards_times_df.columns[1:]:
        reformatted_yards_times_df[col] = reformatted_yards_times_df[col].apply(reformat_time)
    
    normalized_df = reformatted_yards_times_df.copy()
    for col in reformatted_yards_times_df.columns[1:]:
        normalized_df[col] = NCAA_records[col] / reformatted_yards_times_df[col]

    if save_csv == True:
        normalized_df.to_csv("normalized_df.csv", index=False)

    return normalized_df

def create_metrics(normalized_df: pd.DataFrame, save_csv: bool):
    # Assign each event a distance and stroke category
    event_profiles = {
        '50 Y Free': {'distance_score': 1, 'stroke_score': 4},
        '100 Y Free': {'distance_score': 2, 'stroke_score': 4},
        '200 Y Free': {'distance_score': 3, 'stroke_score': 4},
        '500 Y Free': {'distance_score': 4, 'stroke_score': 4},
        '1000 Y Free': {'distance_score': 5, 'stroke_score': 4},
        '1650 Y Free': {'distance_score': 6, 'stroke_score': 4},
        '100 Y Back': {'distance_score': 2, 'stroke_score': 2},
        '200 Y Back': {'distance_score': 3, 'stroke_score': 2},
        '100 Y Breast': {'distance_score': 2, 'stroke_score': 3},
        '200 Y Breast': {'distance_score': 3, 'stroke_score': 3},
        '100 Y Fly': {'distance_score': 2, 'stroke_score': 1},
        '200 Y Fly': {'distance_score': 3, 'stroke_score': 1},
        '200 Y IM': {'distance_score': 3, 'stroke_score': 5},
        '400 Y IM': {'distance_score': 4, 'stroke_score': 5}
    }

    race_metrics_df = pd.DataFrame()
    
    # For each swimmer (row) in our data
    for idx, row in normalized_df.iterrows():
        swimmer_metrics = {}
        
        # Initialize containers for our calculations
        sprint_times = []
        mid_times = []
        distance_times = []
        stroke_performances = {1: [], 2: [], 3: [], 4: [], 5: []}  # For each stroke type
        
        # Process each event
        for event in event_profiles.keys():
            if event in row.index and not pd.isna(row[event]):
                profile = event_profiles[event]
                performance = row[event]
                
                # Categorize by distance, events can be in multiple categories
                if profile['distance_score'] <= 3:
                    sprint_times.append(performance)
                elif profile['distance_score'] in [3, 4]:
                    mid_times.append(performance)
                elif profile['distance_score'] >= 4:
                    distance_times.append(performance)
                
                # Collect stroke-specific performances
                stroke_performances[profile['stroke_score']].append(performance)
        
        # Calculate metrics
        swimmer_metrics['sprint_strength'] = np.mean(sprint_times) if sprint_times else np.nan
        swimmer_metrics['mid_strength'] = np.mean(mid_times) if mid_times else np.nan
        swimmer_metrics['distance_strength'] = np.mean(distance_times) if distance_times else np.nan
        
        # Calculate distance trend, where a positive trend means the swimmer is better at distance
        if sprint_times and distance_times:
            swimmer_metrics['distance_trend'] = np.mean(distance_times) - np.mean(sprint_times)
        else:
            swimmer_metrics['distance_trend'] = np.nan
        
        # Calculate stroke-specific strengths
        for stroke in range(1, 6):
            stroke_name = ['fly', 'back', 'breast', 'free', 'im'][stroke-1]
            swimmer_metrics[f'{stroke_name}_strength'] = (
                np.mean(stroke_performances[stroke]) if stroke_performances[stroke] else np.nan
            )
        
        # Calculate 'stroke versatility' (standard deviation of stroke performances)
        stroke_averages = [np.mean(perfs) for perfs in stroke_performances.values() if perfs]
        swimmer_metrics['stroke_versatility'] = np.std(stroke_averages) if len(stroke_averages) > 1 else np.nan
        
        race_metrics_df = pd.concat([
            race_metrics_df, 
            pd.DataFrame([swimmer_metrics], index=[idx])
        ])

    race_metrics_df.insert(0, 'Name', normalized_df['Name'])

    if save_csv == True:
        race_metrics_df.to_csv("race_metrics_df.csv", index=False)

    return race_metrics_df

# Helper function to reformat times to sec:mili
def reformat_time(time_val):
    if pd.isna(time_val):
        return time_val
    
    time_str = str(time_val)
    
    if ':' in time_str:  # min:sec.mili format
        mins, rest = time_str.split(':')
        secs = float(mins) * 60 + float(rest)
    else:  # already in sec.mili format
        secs = float(time_str)
        
    return secs  

# Prints the rank of every swimmer in each category
def print_rankings(metrics: pd.DataFrame):
    for column in metrics.columns[1:]:
        print(f'\n' + column + ' Rankings:')
        sorted_df = metrics.sort_values(column)[['Name', column]]
        print(sorted_df)

# Runs this file to create the feature df to be clustered
def run_metrics():
    new_df = create_metrics(normalize_to_NCAA_record(make_yards_times_df(make_times_df(False), False), False), True)
    return new_df

