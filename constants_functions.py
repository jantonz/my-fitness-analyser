## Import Modules

import streamlit as st
# Use the warnings module to supress DeprecationWarning around using GroupBy.apply... This will be fixed, particularly before upgrading Pandas to later editions.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import sys
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype
)
import altair as alt
import datetime as dt
import random
from itertools import product
import base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


## Define constants

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


## Create functions

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
        # Remove columns that only have NaN values in them.
        data = data.dropna(axis=1, how='all')
        return data
    else:
        st.error("Error reading .csv extract. Please ensure it has the correct column names and allowable values. If issue persists, contact details in side menu.", icon="🚨")
        sys.exit()


def filter_dataframe(df, units):
    """Adds a UI on top of a dataframe to let users filter columns"""
    modify = st.checkbox("Filter your activities?")
    if not modify:
        return df
    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    try:
        # This try block tries any filter input by the user and if there is any error it informs the user to remove the last filter applied.
        for col in df.columns:
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
    except:
         st.error("Your last filter caused an error please remove.", icon="🚨")


def slider_filter(message,df):
    """Creates a slider between a min and max value.
    Returns a filtered df, number of days on the slider and the minimum selected date."""
    min_value = min(df['Activity Date'].dt.date)
    max_value = max(df['Activity Date'].dt.date)
    if df.empty:
       st.write("No activity data available.")
       st.stop()
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
       st.write("Only one day of activity data available. Please increase filter range.")
       return df, 0, max_value
    

def generate_date_range(start_date, end_date):
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += dt.timedelta(days=1)
    return date_range


def select_slider_filter(message,df):
    """Creates a slider between a min date and a fixed max date value. The fixed max date value is the max date of activities on the dataframe.
    Returns a filtered df, number of days on the slider and the minimum selected date."""
    min_value = min(df['Activity Date'].dt.date)
    max_value = max(df['Activity Date'].dt.date)
    options = generate_date_range(min_value, max_value)
    if df.empty:
       st.write("No activity data available.")
       st.stop()
    if min_value < max_value:
        dates_selection = st.select_slider('%s' % (message),
                                    options =options)
        mask = df['Activity Date'].dt.date.between(dates_selection,max_value)
        time_difference = max_value - dates_selection
        number_of_days_on_slider = time_difference.days
        filtered_df = df[mask]
        return filtered_df, number_of_days_on_slider, dates_selection
    else:
       st.write("Only one day of activity data available. Please increase filter range.")
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



def create_area_chart(df, y_axis, units):
    """Creates a area chart with a dynamic y_axis value"""
    df['Moving Time'] = df['Moving Time (hours)'].apply(format_hours)
    if y_axis == 'Distance':
        y= alt.Y(f'{y_axis}:Q', title = f'{y_axis} ({units[0]})')
    elif y_axis == 'Elevation Gain':
        y= alt.Y(f'{y_axis}:Q', title = f'{y_axis} ({units[1]})')
    elif y_axis == 'Elapsed Time (hours)':
         y= alt.Y(f'{y_axis}:Q', title = 'Number of Activities')
    else:
        y = f'{y_axis}:Q'
    chart = alt.Chart(df).mark_area(opacity=0.8).encode(
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
                    df['Elapsed Time'] = df['Elapsed Time (hours)'].apply(format_hours)
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
                                alt.Tooltip('Elapsed Time:N'),
                                alt.Tooltip('Ratio of Move to Total Time:Q', format=',.3f'),
                                alt.Tooltip('Average Speed:Q', format=',.2f', title = f'Average Speed ({units[2]})'),
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
       

def create_scatter_graph_and_regression(df, metric, units, predictor_1, predictor_2, end_date):
    try:
        if df.empty:
                st.error("No data to display. Ensure there is at least 2 activities within selected filters.", icon="🚨")
                st.stop()
        df['Cumulative Moving Time'] = df['Cumulative Moving Time (hours)'].apply(format_hours)
        if metric == 'Cumulative Distance':
            y= alt.Y(f'{metric}:Q', title = f'{metric} ({units[0]})')
        elif metric == 'Cumulative Elevation Gain':
            y= alt.Y(f'{metric}:Q', title = f'{metric} ({units[1]})')
        else:
            y = alt.Y(f'{metric}:Q', title = f'{metric}')
        df_altair = alt.Chart(df).mark_circle(color='blue').encode(
                                x='Activity Date:T',
                                y= y,
                                color=alt.Color('Activity Type:N',legend={'orient':'top'}, sort=allowable_activities).scale(domain=allowable_activities, range=colours_of_activities),
                                tooltip=[alt.Tooltip('Activity Type:N'),
                                alt.Tooltip('Activity Date:T'),
                                alt.Tooltip('Cumulative Distance:Q', format=',.0f', title = f'Cumulative Distance ({units[0]})'),
                                alt.Tooltip('Cumulative Moving Time:N'),
                                alt.Tooltip('Cumulative Elevation Gain:Q', format=',.0f', title = f'Cumulative Elevation Gain ({units[1]})')]
                            ).interactive()
        combined_chart = df_altair
        r_squared_values = {}
        mae_values = {}
        numbers_of_activities = {}
        predicted_values_activity_type = {}
        for index, activity_type in enumerate(df['Activity Type'].unique()):
            if index >0:
                index += index*15
            data = df[df['Activity Type'] == activity_type]
            number_of_activities = len(data)
            numbers_of_activities[activity_type] = number_of_activities
            # We do not want to include a regression line if there are less than two activities for an activity type
            if number_of_activities < 2:
                break
            X = data['Activity Date'].astype(np.int64).values.reshape(-1, 1)  # Convert dates to numerical values
            y = data[metric].values
            # Initiate linear regression model
            model = LinearRegression()
            # Train linear regression model
            model.fit(X, y)
            y_pred = model.predict(X)
            # Calculate R-squared
            r_squared = r2_score(y, y_pred)
            r_squared = round(r_squared, 2)
            r_squared_values[activity_type] = r_squared
            # Calculate MAE
            mae = mean_absolute_error(y, y_pred)
            mae = round(mae, 2)
            mae_values[activity_type] = mae
            # Predict metric for future dates
            first_date = min(data['Activity Date'].dt.date)
            future_dates = pd.date_range(start=first_date, end=end_date)
            future_dates_num = np.array(future_dates).astype(np.int64).reshape(-1, 1)
            predicted = model.predict(future_dates_num)
            # Add predicted metrics to future dates
            future_df = pd.DataFrame({'Activity Date': future_dates, f'Predicted {metric}': predicted})
            future_df['Activity Date'] = future_df['Activity Date'].dt.strftime('%Y-%m-%d')
            # Match activity types with their corresponding colour (for the trendlines)
            if activity_type == allowable_activities[0]:
                color_of_trendline = colours_of_activities[0]
            elif activity_type == allowable_activities[1]:
                color_of_trendline = colours_of_activities[1]
            elif activity_type == allowable_activities[2]:
                color_of_trendline = colours_of_activities[2]
            elif activity_type == allowable_activities[3]:
                color_of_trendline = colours_of_activities[3]
            chart = alt.Chart(future_df).mark_line(color=color_of_trendline, strokeDash=[1, 2]).encode(
                x='Activity Date:T',
                y= alt.Y(f'Predicted {metric}:Q', title=''),
                tooltip=[alt.Tooltip('Activity Date:T'),
                        alt.Tooltip(f'Predicted {metric}:Q', format=',.0f', title='Predicted Value')]
            ).interactive()
            combined_chart = combined_chart + chart
            # The prediction vertical lines now have to be added to the chart
            list_of_vertical_lines = [predictor_1,predictor_2]
            predicted_values = {}
            for vertical_date in list_of_vertical_lines:
                vertical_line = alt.Chart(pd.DataFrame({'vertical_date': [vertical_date]})).mark_rule(strokeDash=[10, 5], color='red').encode(
                    x= alt.X('vertical_date:T', title = '')
                )
                vertical_date = vertical_date.strftime('%Y-%m-%d')
                predicted_value = np.round(future_df.loc[future_df['Activity Date'] == vertical_date, f'Predicted {metric}'].iloc[0]).astype(int)
                predicted_values[vertical_date] = predicted_value
                label = activity_type + ":\n" + str(predicted_value)
                value_annotation = alt.Chart(pd.DataFrame({'vertical_date': [vertical_date], 'label': label})).mark_text(dx=0, dy=(index), color='black').encode(
                    x='vertical_date:T',
                    text= 'label'
                )
                if index == 0:
                    date_annotation = alt.Chart(pd.DataFrame({'vertical_date': [vertical_date], 'label': [vertical_date]})).mark_text(dx=0, dy=-15, color='black').encode(
                    x='vertical_date:T',
                    text='label'
                    )
                    combined_chart = combined_chart + vertical_line + date_annotation + value_annotation
                else:
                    combined_chart = combined_chart + vertical_line + value_annotation
            predicted_values_activity_type[activity_type] = predicted_values
    except Exception as e:
        st.error("Adjust the configuration so your predictor dates are within the range of the regression line", icon="🚨")
    return combined_chart, numbers_of_activities, r_squared_values, mae_values, predicted_values_activity_type


def write_statistics_of_regression(activity_type, r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric):
    try:
        if metric == 'Cumulative Elevation Gain':
            units = f' {units[1]}'
        elif metric == 'Cumulative Distance':
            units = f' {units[0]}'
        else:
            units = 'hours'
        r_squared = r_squared_values[activity_type]*100
        st.subheader(f"{activity_type}")
        # Predicted values are stored in a nested dict so they need to be unpacked according to the activity type
        keys = predicted_values_activity_type[activity_type].keys()
        for key in keys:
            st.write(key, ": ", predicted_values_activity_type[activity_type][key], units)
        st.caption(f"Number of activities in model calculation: __{numbers_of_activities[activity_type]}__")
        st.caption(f"The model captures about __{r_squared}%__ of the patterns in the data.")
        st.caption(f"On average, the predictions __are off by {mae_values[activity_type]} {units}__ compared to the actual values.")
    except Exception as e:
        # No error message is displayed as the error is handled in the create_scatter_graph_and_regression function that calls this one
        pass


def create_columns_for_statistics(df, predicted_values_activity_type, r_squared_values, mae_values, numbers_of_activities, units, metric):
    activity_types = df['Activity Type'].unique()
    number_of_activity_types = len(activity_types)
    # Depending on the number of activity types being calculated, the statistics are displayed in one of the columns. There is a bit of duplication here that I would like to remove
    if number_of_activity_types == 1:
        write_statistics_of_regression(activity_types[0], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
    elif number_of_activity_types == 2:
        col1, col2 = st.columns(2)
        with col1:
            write_statistics_of_regression(activity_types[0], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col2:
            write_statistics_of_regression(activity_types[1], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
    elif number_of_activity_types == 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            write_statistics_of_regression(activity_types[0], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col2:
            write_statistics_of_regression(activity_types[1], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col3:
            write_statistics_of_regression(activity_types[2], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
    elif number_of_activity_types == 4:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            write_statistics_of_regression(activity_types[0], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col2:
            write_statistics_of_regression(activity_types[1], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col3:
            write_statistics_of_regression(activity_types[2], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)
        with col4:
            write_statistics_of_regression(activity_types[3], r_squared_values, mae_values, numbers_of_activities, predicted_values_activity_type, units, metric)

            

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