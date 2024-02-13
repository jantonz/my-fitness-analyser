import streamlit as st
import pandas as pd
import sys
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import altair as alt
import datetime as dt
import random
from itertools import product
import base64


# These columns are the minimum required for this script to function.
minimum_columns_from_excel_extract = ['Activity Date','Activity Type','Elapsed Time', 'Moving Time', 'Distance', 'Elevation Gain']
# These columns represent the full suite of possible columns to include.
full_columns_from_excel_extract = ['Activity ID', 'Activity Date','Activity Name','Activity Type','Activity Description','Elapsed Time', 'Moving Time', 'Distance', 'Elevation Gain', 'Elevation Loss', 'Elevation Low', 'Elevation High']

column_mapping = {'Elapsed Time': 'Elapsed Time (hours)',
                  'Moving Time': 'Moving Time (hours)',
                  'Distance': 'Distance'}

elevation_columns = ['Elevation Gain', 'Elevation Loss', 'Elevation Low', 'Elevation High']

speed_column = ['Average Speed']

distance_column = ['Distance']

allowable_activities = ['Run','Ride','Swim','Walk']

colours_of_activities = ['Orange', 'Green', 'Purple', 'Blue']

periods_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']

colours_of_periods_of_day = ['#306182', '#768b99', '#5ebeff', '#9c5959']

metre_foot_conv = 3.28084

metre_mile_conv = 0.0006213712

final_order_of_columns_for_table_display = ['Activity ID', 'Activity Name', 'Activity Type', 'Activity Date', 'Elapsed Time (hours)', 'Moving Time (hours)', 'Ratio of Move to Total Time', 'Distance', 'Average Speed', 'Elevation Gain', 'Elevation Loss', 'Elevation Low', 'Elevation High']


def remove_and_rename(df, target_column_name, column_to_remove):
    """Manages instances where there are duplicate column names"""
    if target_column_name in df.columns:
        df = df.drop(column_to_remove, axis=1, errors='ignore')
        df = df.rename(columns={target_column_name: column_to_remove})
    else:
         pass
    return df

@st.cache_data
def load_data(file, units):
    """Loads, cleans and transforms input .csv data.
    Data is stored in the cache."""
    data = pd.read_csv(file)
    data = remove_and_rename(data, 'Distance.1', 'Distance')
    data = remove_and_rename(data, 'Elapsed Time.1', 'Elapsed Time')
    if set(minimum_columns_from_excel_extract).issubset(data.columns):
        for col in full_columns_from_excel_extract:
            if col not in data.columns:
                data[col] = None  
        data = data[full_columns_from_excel_extract]
        data.rename(columns=column_mapping, inplace=True)
        # Add any activities in the "hike" category to the "walk" category
        data['Activity Type'] = data['Activity Type'].replace('Hike', 'Walk')
        data = data[data['Activity Type'].isin(allowable_activities)]
        data['Activity Date'] = pd.to_datetime(data['Activity Date'], format='mixed', dayfirst=True)
        # Remove any rows that has an NaN in either the Elapsed or Moving Time columns.
        data = data.dropna(subset=['Elapsed Time (hours)', 'Moving Time (hours)'])
        data['Elapsed Time (hours)'] = round((data['Elapsed Time (hours)']) / 60,2)
        data['Moving Time (hours)'] = round((data['Moving Time (hours)']) / 60,2)
        data['Ratio of Move to Total Time'] = data['Moving Time (hours)'] / data['Elapsed Time (hours)']
        # Fill in NaN values with 0 in the elevation columns.
        # Handling the different unit types to convert the distance, speed and elevation columns to the correct values using conversion factors defined on this script.
        if units[0] == "km":
            data['Distance'] = round((data['Distance'])/1000,2)
            data['Average Speed'] = round(data['Distance'] / (data['Moving Time (hours)']/60),2)
            data[elevation_columns] = data[elevation_columns].round(2)
        else:
            data['Distance'] = round((data['Distance']*metre_mile_conv),2)
            data['Average Speed'] = round(data['Distance'] / (data['Moving Time (hours)']/60),2)
            data[elevation_columns] = data[elevation_columns].apply(lambda x: x * metre_foot_conv).round(2)
        data['Elapsed Time (hours)'] = round((data['Elapsed Time (hours)']) / 60,2)
        data['Moving Time (hours)'] = round((data['Moving Time (hours)']) / 60,2)
        data = data[final_order_of_columns_for_table_display]
        return data
    else:
        st.markdown('''**:blue[Error reading .csv extract. Please ensure it has the correct column names and allowable values. If issue persists, contact details in side menu.]**''')
        sys.exit()


def filter_dataframe(df, units):
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    """
    modify = st.checkbox("Filter your activities?")
    if not modify:
        return df
    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter activities on:", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    return df


def slider_filter(message,df):
    """Creates a slider between a min and max value.
    Returns a filtered df, number of days on the slider and the minimum selected date."""
    min_value = min(df['Activity Date'].dt.date)
    max_value = max(df['Activity Date'].dt.date)
    if min_value < max_value:
        dates_selection = st.slider('%s' % (message),
                                    min_value = min(df['Activity Date'].dt.date),
                                    max_value = max(df['Activity Date'].dt.date),
                                    value =(min(df['Activity Date'].dt.date),max(df['Activity Date'].dt.date)), step= dt.timedelta(days=1))
        mask = df['Activity Date'].dt.date.between(*dates_selection)
        time_difference = dates_selection[1] - dates_selection[0]
        number_of_days_on_slider = time_difference.days
        filtered_df = df[mask]
        return filtered_df, number_of_days_on_slider, dates_selection[0]
   
    else:
       st.write("Only one day of activity data available")
       return df, 0, max_value
        

def format_hours(hours):
    """Format an int to become a string with an hour and min section."""
    hours_int = int(hours)
    minutes = int((hours - hours_int) * 60)
    minutes = round(minutes, 2)
    return f"{hours_int}h {minutes}m"


def format_metric_values(df, merged_df, activity_type, metric, units):
    """Some string manipulation to create the format for overall and change metrics."""
    formatted_value = "{:,.2f} ".format(df.at[df[df['Activity Type'] == activity_type].index[0], metric])
    formatted_change = "{:.2f}%".format(merged_df.at[merged_df[merged_df['Activity Type'] == activity_type].index[0], metric])
    formatted_value = formatted_value + units
    return formatted_value, formatted_change


def create_metrics(df, merged_df, activity_type, first_column, second_column, third_column, four_column, units):
    """Creates the metrics on the dashboard with 4 columns and a metric for each."""
    random_int = random.randint(2, 100)
    col1= first_column
    col2 = second_column
    col3 = third_column
    col4 = four_column
    col1, col2, col3, col4 = st.columns(4, gap="small")
    formatted_activity_count_value = "{:,} ".format(df.at[df[df['Activity Type'] == activity_type].index[0], 'Activity Date'])
    formatted_activity_count_gain = "{:.2f}%".format((merged_df.at[merged_df[merged_df['Activity Type'] == activity_type].index[0], 'Activity Date']))
    formatted_hour_value = df.at[df[df['Activity Type'] == activity_type].index[0], 'Moving Time (hours)']
    formatted_hour_gain = "{:.2f}%".format((merged_df.at[merged_df[merged_df['Activity Type'] == activity_type].index[0], 'Moving Time (hours)']))
    formatted_distance_value, formatted_distance_change = format_metric_values(df, merged_df, activity_type, 'Distance', units[0])
    formatted_elevation_value, formatted_elevation_change = format_metric_values(df, merged_df, activity_type, 'Elevation Gain', units[1])
    col1.metric('Total Number of Activities', formatted_activity_count_value, formatted_activity_count_gain)
    col2.metric('Total Distance', formatted_distance_value, formatted_distance_change)
    col3.metric('Total Moving Time', formatted_hour_value, formatted_hour_gain)
    if activity_type == 'Swim':
        col4.metric('Total Elevation Gain', f"{random_int} Niagara Falls", "you can do better")
    else:
        col4.metric('Total Elevation Gain', formatted_elevation_value, formatted_elevation_change)



def create_bar_chart(df, y_axis, units):
    """Creates a bar chart with a dynamic y_axis value"""
    df['Moving Time'] = df['Moving Time (hours)'].apply(format_hours)
    if y_axis == 'Distance':
        y= alt.Y(f'{y_axis}:Q', title = f'{y_axis} ({units[0]})')
    elif y_axis == 'Elevation Gain':
        y= alt.Y(f'{y_axis}:Q', title = f'{y_axis} ({units[1]})')
    elif y_axis == 'Elapsed Time (hours)':
         y= alt.Y(f'{y_axis}:Q', title = 'Number of Activities')
    else:
        y = f'{y_axis}:Q'
    chart = alt.Chart(df).mark_bar(opacity=0.8).encode(
          x='Activity Date:T',
          y= y,
          color=alt.Color('Activity Type:N', legend=None).scale(domain=allowable_activities, range=colours_of_activities),
          tooltip=[alt.Tooltip('Activity Type:N'),
                   alt.Tooltip('Activity Date:T', title='Week Beginning'),
                   alt.Tooltip('Elapsed Time (hours):Q', title='Number of Activities'),
                   alt.Tooltip('Moving Time:N'),
                   alt.Tooltip('Distance:Q', format=',.2f', title = f'Distance ({units[0]})'),
                   alt.Tooltip('Elevation Gain:Q', format=',.2f', title = f'Elevation Gain ({units[1]})')]).interactive()
    return chart


def create_scatter_graph(df, units):
                    """Creates a scatter graph of distance and time"""
                    df['Moving Time'] = df['Moving Time (hours)'].apply(format_hours)
                    bind_checkbox = alt.binding_checkbox(name='Scale point size by elevation gain? ')
                    param_checkbox = alt.param(bind=bind_checkbox)
                    chart = alt.Chart(df).mark_point().encode(
                        x= alt.X('Distance:Q', title = f'Distance ({units[0]})'),
                        y='Moving Time (hours):Q',
                        color=alt.Color('Activity Type:N',legend=None, sort= allowable_activities).scale(domain=allowable_activities, range=colours_of_activities),
                        tooltip=[alt.Tooltip('Activity Date:T'),
                                alt.Tooltip('Activity Name:N'),
                                alt.Tooltip('Activity Type:N'),
                                alt.Tooltip('Distance:Q', format=',.2f', title = f'Distance ({units[0]})'),
                                alt.Tooltip('Moving Time:N'),
                                alt.Tooltip('Ratio of Move to Total Time:Q', format=',.3f'),
                                alt.Tooltip('Average Speed:Q', format=',.2f', title = f'Distance ({units[2]})'),
                                alt.Tooltip('Elevation Gain:Q', format=',.2f', title = f'Elevation Gain ({units[1]})'),
                                alt.Tooltip('Elevation High:Q', format=',.2f', title = f'Elevation High ({units[1]})')],
                        size=alt.condition(param_checkbox,
                        'Elevation Gain:Q',
                        alt.value(25), legend=None
                    )).add_params(
                    param_checkbox
                    ).interactive()
                    return chart



def create_average_speed_line_chart(df, units):
                start_date = df['Activity Date'].dt.date.min()
                end_date = df['Activity Date'].dt.date.max()
                date_range_df = pd.DataFrame({
                                        'Date': pd.date_range(start=start_date, end=end_date, freq='D')})     
                date_range_df['Date'] = date_range_df['Date'].dt.strftime('%Y-%m-%d')
                # Create a list of activity types
                activity_types = df['Activity Type'].unique()
                # Create copy of our filtered dataframe
                df_copy = df.copy()
                df_copy = df_copy[['Activity Date', 'Activity Type', 'Average Speed']]
                df_copy = df_copy.rename(columns={'Activity Date': 'Date'})
                df_copy['Date'] = df_copy['Date'].dt.strftime('%Y-%m-%d')
                # Take the average, average speed per day per activity type in case there are >1 of the same activity type in a day
                df_copy = df_copy.groupby(['Date', 'Activity Type']).agg({'Average Speed': 'mean'}).reset_index()
                # Create a Cartesian product of dates and activity types
                date_activity_combinations = list(product(date_range_df['Date'], activity_types))
                date_activity_df = pd.DataFrame(date_activity_combinations, columns=['Date', 'Activity Type'])
                merged_df = pd.merge(date_activity_df, df_copy, on=['Date', 'Activity Type'], how='left')
                merged_df['Average Speed'] = merged_df.groupby('Activity Type')['Average Speed'].transform(lambda x: x.expanding().mean())
                chart = alt.Chart(merged_df).mark_line().encode(
                    x='Date:T',
                    y= alt.Y('Average Speed:Q', title = f'Average Speed ({units[2]})'),
                    color=alt.Color('Activity Type:N',legend=None).scale(domain=allowable_activities, range=colours_of_activities),
                    tooltip=[alt.Tooltip('Date:T'),
                            alt.Tooltip('Activity Type:N'),
                            alt.Tooltip('Average Speed:Q', format=',.4f', title = f'Average Speed ({units[2]})')]
                ).interactive()
                return chart



def create_mark_bar_weighted_average(df, units):
                """Creates a horizontal bar chart for each activity split by each period of the day."""
                chart = alt.Chart(df).mark_bar().encode(
                        x= alt.X('Weighted Avg Speed:Q', title = f'Weighted Average Speed ({units[2]})'),
                        y= alt.Y('Period of Day:N', sort=periods_of_day, title=None),
                        color=alt.Color('Period of Day:N',legend=None).scale(domain=periods_of_day, range=colours_of_periods_of_day),
                        row= alt.Row('Activity Type:N', sort=allowable_activities),
                        tooltip=[alt.Tooltip('Period of Day:N'),
                                alt.Tooltip('Activity Type:N'),
                                alt.Tooltip('Weighted Avg Speed:Q', format=',.4f', title = f'Weighted Avg Speed ({units[2]})')]
                        ).interactive()
                return chart


def categorise_period(hour):
       """Categorise the periods of the day"""
       if 0 <= hour < 5:
        return 'Night'
       elif 5 <= hour < 11:
        return 'Morning'
       elif 11 <= hour < 17:
        return 'Afternoon'
       elif 17 <= hour < 22:
        return 'Evening'
       else:
           return 'Night'
       

# Not used in this release. This code creates a time series of the weighted average speed, however, errors with the cumsum() method proved it to be unusable in its current state.
                # weighted_avg_speed_df = slider_filtered_df.copy()
                # weighted_avg_speed_df = weighted_avg_speed_df.rename(columns={'Activity Date': 'Date'})
                # weighted_avg_speed_df['Cumulative Distance'] = weighted_avg_speed_df.groupby(['Activity Type'])['Distance'].cumsum()
                # # Strange values here ^
                # weighted_avg_speed_df['Date'] = weighted_avg_speed_df['Date'].dt.strftime('%Y-%m-%d')
                # weighted_avg_speed_df['Speed x Distance'] = weighted_avg_speed_df['Distance'] * weighted_avg_speed_df['Average Speed']
                # weighted_avg_speed_df = weighted_avg_speed_df.groupby(['Date', 'Activity Type']).agg({'Speed x Distance': 'sum', 'Cumulative Distance': 'last'}).reset_index()
                # #Use Cartersian product of dates and activity types from previous tab.
                # start_date = slider_filtered_df['Activity Date'].dt.date.min()
                # end_date = slider_filtered_df['Activity Date'].dt.date.max()
                # date_range = m_c.pd.date_range(start=start_date, end=end_date, freq='D')
                # date_range_df = m_c.pd.DataFrame({
                #                         'Date': m_c.pd.date_range(start=start_date, end=end_date, freq='D')})
                # date_range_df['Date'] = date_range_df['Date'].dt.strftime('%Y-%m-%d')
                # # Create a list of activity types
                # activity_types = slider_filtered_df['Activity Type'].unique()
                # date_activity_combinations = list(m_c.product(date_range_df['Date'], activity_types))
                # date_activity_df = m_c.pd.DataFrame(date_activity_combinations, columns=['Date', 'Activity Type'])
                # merged_df = m_c.pd.merge(date_activity_df, weighted_avg_speed_df, on=['Date', 'Activity Type'], how='left')

                # def calculate_weighted_avg(df):
                #     dict_of_dataframes = {}
                #     for activitytype in df['Activity Type'].unique():
                #         group = df[df['Activity Type'] == activitytype]
                #         filtered_df = group[group['Cumulative Distance'].notnull()]
                #         # Apply the condition to replace the value if it's more than 5 times the size of the previous row
                #         mask = (filtered_df['Cumulative Distance'] > 5 * filtered_df['Cumulative Distance'].shift(1))
                #         filtered_df.loc[mask, 'Cumulative Distance'] = filtered_df['Cumulative Distance'].shift(1)
                #                         # Merge the changes back to the original DataFrame
                #         group = m_c.pd.merge(group, filtered_df[['Date', 'Cumulative Distance']], on='Date', how='left', suffixes=('', '_filtered'))
                #                         # Fill NaN values in the original 'Cumulative Distance' column with the filtered values
                #         group['Cumulative Distance'] = group['Cumulative Distance_filtered'].combine_first(group['Cumulative Distance'])
                #                         # Drop the extra columns used for filtering                 
                #         group.drop(['Cumulative Distance_filtered'], axis=1, inplace=True)
                #         non_null_rows = group['Speed x Distance'].notnull()             
                #                         #     # Calculate weighted average speed only for non-null rows
                #         group.loc[non_null_rows, 'Weighted Average Speed'] = group.loc[non_null_rows, 'Speed x Distance'].cumsum() / group.loc[non_null_rows, 'Cumulative Distance']
                #         group['Weighted Average Speed'] = group['Weighted Average Speed'].ffill()
                #         dict_of_dataframes[activitytype] = group
                #     dfs = m_c.pd.concat(dict_of_dataframes.values(), ignore_index=True)
                #     return dfs
           


                # test = calculate_weighted_avg(merged_df)
                # average_speed_line_chart = m_c.create_line_chart(test, units, selection_global)
                # m_c.st.altair_chart(activity_selector & average_speed_line_chart, use_container_width=True)