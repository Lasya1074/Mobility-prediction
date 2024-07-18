#Importing the libraries
import numpy as np
import pandas as pd
import glob
import os


# Define mode names and IDs for different transportation modes
mode_names = ['walk', 'bike', 'bus', 'car', 'subway', 'train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}  # Assign unique IDs to each mode


# Function to read .plt files into a DataFrame
def read_plt(plt_file):

    """
    Function to read .plt files into a DataFrame.

    Parameters:
    - plt_file (str): Path to the .plt file.

    Returns:
    - points (DataFrame): DataFrame containing GPS points.
    """
    
    # Read the .plt file skipping the first 6 rows and without a header
    points = pd.read_csv(plt_file, skiprows=6, header=None)
    # Rename columns for better readability
    points.rename(inplace=True, columns={0: 'lat', 1: 'lon', 3: 'alt', 5: 'day', 6: 'hour'})
    date_format = '%Y-%m-%d %H:%M:%S'
    # Combine 'day' and 'hour' columns to create a 'time' column of datetime format
    points['time'] = pd.to_datetime(points['day'] + ' ' + points['hour'], format=date_format)
    # Drop unnecessary columns
    points.drop(inplace=True, columns=[2, 4, 'day', 'hour'])
    return points


# Function to read labels from a file
def read_labels(labels_file):

    """
    Function to read labels from a file.

    Parameters:
    - labels_file (str): Path to the labels file.

    Returns:
    - labels (DataFrame): DataFrame containing labeled segments.
    """
     
    # Read labels file skipping the first row and using whitespace as separator
    labels = pd.read_csv(labels_file, skiprows=1, header=None, sep='\s+')
    # Combine date and time for start time
    labels['start_time'] = pd.to_datetime(labels[0] + ' ' + labels[1])
    # Combine date and time for end time
    labels['end_time'] = pd.to_datetime(labels[2] + ' ' + labels[3])    
    # Rename columns for better readability
    labels.rename(inplace=True, columns={4: 'label'})
    # Drop unnecessary columns
    labels.drop(inplace=True, columns=[0, 1, 2, 3])
    # Map labels to their corresponding mode IDs
    labels['label'] = [mode_ids[i] for i in labels['label']]
    return labels


# Function to apply labels to GPS points
def apply_labels(points, labels):

    """
    Function to apply labels to GPS points.

    Parameters:
    - points (DataFrame): DataFrame containing GPS points.
    - labels (DataFrame): DataFrame containing labeled segments.
    """
     
    # Find the index where each point's time is less than the label's end time
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    # Condition to check points that have no corresponding label
    no_label_condition = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    # Assign labels to points
    points['label'] = labels['label'].iloc[indices].values
     # Set label to 0 where no label applies
    points.loc[no_label_condition, 'label'] = 0 


# Function to read user data from a folder containing trajectory files and a labels file
def read_user(user_folder):

    """
    Function to read user data from a folder containing trajectory files and a labels file.

    Parameters:
    - user_folder (str): Path to the user's folder.

    Returns:
    - df (DataFrame): Concatenated DataFrame of user's trajectory data.
    """
    
    # Get all .plt files in folder
    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    # Concatenate all .plt files into a single DataFrame
    df = pd.concat([read_plt(f) for f in plt_files])
    # Path to the labels file
    labels_file = os.path.join(user_folder, 'labels.txt')
    # Check if labels file exists
    if os.path.exists(labels_file):
        # Read and parse labels file
        labels = read_labels(labels_file)
        # Apply labels to the DataFrame
        apply_labels(df, labels)  
    else:
        # If no labels file, set label to 0 for all points
        df['label'] = 0  
    return df


# Function to correct labels by adjusting IDs (subtracting 1 from non-zero labels)
def correct_label(df_user):

    """
    Function to correct labels by adjusting IDs (subtracting 1 from non-zero labels).

    Parameters:
    - df_user (DataFrame): DataFrame containing user's trajectory data.

    Returns:
    - df_user (DataFrame): DataFrame with corrected labels.
    """
        
    # Copy DataFrame excluding points with label 0
    df_user = df_user[df_user['label'] != 0].copy()
    # Adjust label IDs by subtracting 1
    if len(df_user) != 0:
        df_user['label'] = df_user['label'] - 1  
    return df_user


# Function to remove duplicate GPS points based on time
def remove_duplication(df_user):

    """
    Function to remove duplicate GPS points based on time.

    Parameters:
    - df_user (DataFrame): DataFrame containing user's trajectory data.

    Returns:
    - df_user (DataFrame): DataFrame with duplicate points removed.
    """
    
    # Drop duplicates based on 'time' column, Keeping the first occurrence
    df_user = df_user.drop_duplicates(subset=['time'], keep='first')  
    return df_user


# Function to segment user trajectory based on time differences
def segment_user_trajectory(user_folder, df_user, threshold_timediff=(60 * 2)):

    """
    Function to segment user trajectory based on time differences.

    Parameters:
    - user_folder (str): Path to the user's folder.
    - df_user (DataFrame): DataFrame containing user's trajectory data.
    - threshold_timediff (int): Threshold time difference in seconds to segment trajectories.

    Returns:
    - df_modified (DataFrame): DataFrame with segmented user trajectory.
    """
    
    # Extract user name from folder path
    user_name = user_folder.split('/')[-1] 
    tid = 0
    modified_groups = []
    # Group data by label
    for segmentor, group in df_user.groupby('label'):
        # Sort group by time
        group = group.sort_values(by='time', ascending=True)  
        group.index = range(len(group))
        group['tid'] = 'tid'
        # Calculate time differences between consecutive points
        group['time_diff'] = group['time'].diff().dt.total_seconds()  
        group['time_diff'] = group['time_diff'].fillna(0)
        # Find indices where time difference exceeds threshold
        indices = list((group[group['time_diff'] > threshold_timediff]).index)  
        if indices:
            if indices[-1] <= len(group) - 1:
                indices.append(len(group))
            start = 0
            for index in indices:
                # Assign trajectory ID (tid) to segments
                group.loc[start:(index - 1), 'tid'] = f'{user_name}_{tid}'
                start = index
                tid += 1
        else:
            # Assign single trajectory ID if no time difference exceeds threshold
            group['tid'] = f'{user_name}_{tid}'
            tid += 1
        modified_groups.append(group)
        # Concatenate modified groups
    df_modified = pd.concat(modified_groups)
    return df_modified


# Function to partition user trajectory based on maximum number of points
def partition_user_trajectory(df_user_segment, max_point=100):

    """
    Function to partition user trajectory based on maximum number of points.

    Parameters:
    - df_user_segment (DataFrame): DataFrame containing segmented user trajectory.
    - max_point (int): Maximum number of points per partition.

    Returns:
    - df_modified (DataFrame): DataFrame with partitioned user trajectory.
    """
    
    modified_groups = []
    # Group data by trajectory ID (tid)
    for tid, group in list(df_user_segment.groupby('tid')):
        partition = 0
        group = group.sort_values(by='time', ascending=True)
        group.index = range(len(group))
        group['partition'] = 'partition'
        if len(group) > max_point:
            for i in range(0, len(group), max_point):
                if (len(group) - 1) - i + 1 >= max_point:
                    # Assign partition ID to segments with maximum points
                    group.loc[i:(i + max_point - 1), 'partition'] = f'{tid}_{partition}'
                    partition += 1
                else:
                    # Assign partition ID to remaining points
                    group.loc[i:, 'partition'] = f'{tid}_{partition}'
                    partition += 1
        else:
            # Assign single partition ID if group size is less than maximum points
            group['partition'] = f'{tid}_{partition}'
            partition += 1
        modified_groups.append(group)
        # Concatenate modified groups
    df_modified = pd.concat(modified_groups)
    df_modified.index = range(len(df_modified))
    return df_modified


# Function to remove partitions with fewer than a threshold number of points
def remove_unsatisfied_partition(df_user_partition, threshold_num_points=20):

    """
    Function to remove partitions with fewer than a threshold number of points.

    Parameters:
    - df_user_partition (DataFrame): DataFrame containing partitioned user trajectory.
    - threshold_num_points (int): Threshold number of points per partition.

    Returns:
    - satisfied_df_user (DataFrame): DataFrame with satisfied partitions.
    """
    
    # Count number of points in each partition
    partition_counts = df_user_partition['partition'].value_counts().sort_index()
    # Filter partitions that meet the threshold number of points
    satisfied_partition = partition_counts[partition_counts >= threshold_num_points].index
    satisfied_df_user = df_user_partition[df_user_partition['partition'].isin(satisfied_partition)]
    return satisfied_df_user

# Function to calculate distance between two points given their coordinates and altitude
def calculate_distance(lat1, lon1, alt1, lat2, lon2, alt2):

    """
    Function to calculate distance between two points given their coordinates and altitude.

    Parameters:
    - lat1, lon1, alt1 (float): Latitude, longitude, and altitude of the first point.
    - lat2, lon2, alt2 (float): Latitude, longitude, and altitude of the second point.

    Returns:
    - distance (float): Distance between the two points in meters.
    """
    
    R = 6371  # Radius of the Earth in kilometers
    lat1_rad = np.deg2rad(lat1)  #Earth Radius and Conversion to Radians
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    #Haversine Formula for Surface Distance
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    surface_distance = R * c * 1000  # Convert distance to meters
    dalt = alt2 - alt1  # Calculate altitude difference
    #Calculate Total 3D Distance
    distance = np.sqrt(surface_distance ** 2 + dalt ** 2)
    return distance


# Function to calculate bearing and pitch between two points given their coordinates and altitude
def calculate_bearing_and_pitch(lat1, lon1, alt1, lat2, lon2, alt2):

    """
    Function to calculate bearing and pitch between two points given their coordinates and altitude.

    Parameters:
    - lat1, lon1, alt1 (float): Latitude, longitude, and altitude of the first point.
    - lat2, lon2, alt2 (float): Latitude, longitude, and altitude of the second point.

    Returns:
    - bearing (float): Bearing from the first point to the second point in degrees.
    - pitch (float): Pitch from the first point to the second point in degrees.
    """
    
    #Convert Differences in Longitude to Radians:
    delta_lon = np.radians(lon2 - lon1)
    #Convert Latitudes to Radians
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2) 
    #Calculate y and x for the Bearing Formula:
    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)
    bearing = np.degrees(np.arctan2(y, x))  #Calculate Bearing in Degrees
    bearing = (bearing + 360) % 360  # Normalize bearing to be within [0, 360)
    #Differences in Latitude and Longitude in Radians
    dlat = lat2_rad - lat1_rad
    dlon = delta_lon
    #Haversine Formula for Horizontal Distance
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    horizontal_distance = 6371 * c * 1000  # Convert distance to meters
    # Calculate Pitch
    dalt = (alt2 - alt1)  # Calculate altitude difference
    pitch = np.degrees(np.arctan2(dalt, horizontal_distance))  #Calculate Pitch in Degrees
    return bearing, pitch


# Function to create additional attributes (distance, speed, acceleration, bearing, pitch) for each point
def create_attribute(df_user):

    """
    Function to create additional attributes (distance, speed, acceleration, bearing, pitch) for each point in user trajectory data.

    Parameters:
    - df_user (DataFrame): DataFrame containing user trajectory data with columns 'lat', 'lon', 'alt', 'time', 'partition'.

    Returns:
    - df_engineering (DataFrame): DataFrame with added attributes ('distance', 'speed', 'acceleration', 'bearing', 'pitch').
    """
    
    modified_groups = []
    for partition, df in df_user.groupby('partition'):
        # Convert altitude from feet to meters
        df['alt'] = df['alt'] * 0.3048  
        df['time_diff'] = df['time'].diff().dt.total_seconds()
        # Calculate time differences
        df['distance'] = calculate_distance(df['lat'].shift().values, df['lon'].shift().values, df['alt'].shift().values, 
                                            df['lat'].values, df['lon'].values, df['alt'].values)  # Calculate distance
        # Calculate speed
        df['speed'] = df['distance'] / df['time_diff']  
        df.fillna(0, inplace=True)
        # Calculate acceleration
        df['acceleration'] = df['speed'].diff() / df['time_diff']
        #calculation of bearing and pitch
        df['bearing'], df['pitch'] = calculate_bearing_and_pitch(df['lat'].shift().values, df['lon'].shift().values, df['alt'].shift().values,
                                                                 df['lat'].values, df['lon'].values, df['alt'].values)  # Calculate bearing and pitch
        df.fillna(0, inplace=True)
        modified_groups.append(df)
    df_engineering = pd.concat(modified_groups)
    return df_engineering



# Function to read all users' data from a main folder containing multiple user folders
def read_all_users(folder):

    """
    Function to read all users' data from a main folder containing multiple user folders of trajectory data.

    Parameters:
    - folder (str): Path to the main folder containing subfolders with user data.

    Returns:
    - df_all_users (DataFrame): Combined DataFrame containing preprocessed trajectory data from all users.
    """
    
    subfolders = os.listdir(folder)  # List all subfolders in main folder
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        user_folder = os.path.join(folder, sf)  # Get path to user folder
        df_user = read_user(user_folder)  # Read user data
        df_user = correct_label(df_user)  # Correct labels
        if len(df_user) != 0:
            df_user = remove_duplication(df_user)  # Remove duplicate points
            df_user_segment = segment_user_trajectory(user_folder, df_user)  # Segment user trajectory
            df_user_partition = partition_user_trajectory(df_user_segment)  # Partition user trajectory
            satisfied_df_user = remove_unsatisfied_partition(df_user_partition)  # Remove unsatisfied partitions
            if len(satisfied_df_user) != 0:
                df_user_engineering = create_attribute(satisfied_df_user)  # Create additional attributes
                df_user_engineering['user'] = int(sf)  # Add user ID column
                dfs.append(df_user_engineering)
    return pd.concat(dfs)


# Function to save preprocessed data to a CSV file
def save_preprocessed_data(df, output_file):

    """
    Function to save preprocessed data DataFrame to a CSV file.

    Parameters:
    - df (DataFrame): Preprocessed data DataFrame to be saved.
    - output_file (str): File path or name of the output CSV file.

    Returns:
    - None
    """
    
    df.to_csv(output_file, index=False)  # Save DataFrame to CSV file without index


# Main function to execute the preprocessing pipeline
def main():

    """
    Main function to execute the preprocessing pipeline for Geolife trajectory data.

    Parameters:
    - None

    Returns:
    - None
    """
    
    data_folder = '/content/dataset/Geolife Trajectories 1.3/Data'  # Path to main data folder
    output_file = 'preprocessed_geolife_data.csv'  # Output CSV file name
    
    # Read and process all user data
    df = read_all_users(data_folder)
    
    # Save the preprocessed data to a CSV file
    save_preprocessed_data(df, output_file)
    
    print(f'Preprocessed data saved to {output_file}')  # Print confirmation message



# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()


