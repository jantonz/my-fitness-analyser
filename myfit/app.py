"""Frontend for myfit analysis."""

import json
import operator
import zipfile
from functools import reduce

import constants_functions as m_c
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_gsheets import GSheetsConnection


def transform_df_sport(df_sport):
    # Create a copy of the dataframe to avoid modifying the original
    df = df_sport.copy()

    # Convert 'Time' to the desired format
    df["Activity Date"] = df["Time"].dt.strftime("%b %d, %Y, %I:%M:%S %p")
    # Map 'Key' to 'Activity Type'
    activity_type_map = {
        "basketball": "Basket",
        "free_training": "Basket",
        "high_interval_training": "HIIT",
        "outdoor_hiking": "Hike",
        "outdoor_riding": "Ride",
        "outdoor_running": "Run",
        "pool_swimming": "Swim",
        # Add more mappings as needed
    }
    df["Activity Type"] = df["Key"].map(activity_type_map)

    # Convert 'duration' to 'Elapsed Time' and 'Moving Time'
    df["Elapsed Time"] = df["duration"]
    df["Moving Time"] = df["duration"]  # Assuming Moving Time is the same as Elapsed Time

    # Convert 'distance' to float and divide by 1000 to get kilometers
    df["Distance"] = pd.to_numeric(df["distance"], errors="coerce") / 1000
    df["Distance.1"] = pd.to_numeric(df["distance"], errors="coerce")

    # Convert 'Elapsed Time' to float for 'Elapsed Time.1'
    df["Elapsed Time.1"] = df["Elapsed Time"].astype(float)

    # Add placeholder columns for elevation data (not present in original data)
    df["Elevation Gain"] = 0
    df["Elevation Low"] = 0
    df["Elevation High"] = 0

    # Select and reorder columns
    columns = [
        "Activity Date",
        "Activity Type",
        "Elapsed Time",
        "Distance",
        "Elapsed Time.1",
        "Moving Time",
        "Distance.1",
        "Elevation Gain",
        "Elevation Low",
        "Elevation High",
    ]

    df_transformed = df[columns]

    # Set data types
    df_transformed = df_transformed.astype(
        {
            "Activity Date": "datetime64[ns]",
            "Activity Type": "object",
            "Elapsed Time": "int64",
            "Distance": "object",
            "Elapsed Time.1": "float64",
            "Moving Time": "int64",
            "Distance.1": "float64",
            "Elevation Gain": "float64",
            "Elevation Low": "float64",
            "Elevation High": "float64",
        },
        errors="ignore",
    )

    return df_transformed


def handle_file_upload(uploaded_file, password):
    sport_filename = f"{uploaded_file.name[:29]}_hlth_center_sport_record.csv"
    fitness_filename = f"{uploaded_file.name[:29]}_hlth_center_fitness_data.csv"
    pwd = password.encode()
    zfile = zipfile.ZipFile(uploaded_file)
    df_sport = pd.read_csv(zfile.open(sport_filename, mode="r", pwd=pwd))
    df_weight = pd.read_csv(zfile.open(fitness_filename, mode="r", pwd=pwd))
    return df_sport, df_weight


def upload_uploaded_file_to_gsheets(uploaded_file, password):
    # Read dataset the first time
    df_sport, df_weight = handle_file_upload(uploaded_file, password)

    # Upload dataset to GSheets
    conn = st.connection("gsheets", type=GSheetsConnection)

    df_sport = conn.update(
        worksheet="sport",
        data=df_sport,
    )
    df_weight = conn.update(
        worksheet="weight",
        data=df_weight,
    )
    return df_sport, df_weight


def load_data_from_gsheets():
    # Download dataset from GSheets
    conn = st.connection("gsheets", type=GSheetsConnection)

    df_sport = conn.read(worksheet="sport")
    df_weight = conn.read(worksheet="weight")
    return df_sport, df_weight


@st.cache_data
def load_data(df_sport, df_weight):
    # Read the data

    df_weight["Time"] = pd.to_datetime(df_weight["Time"], unit="s")
    for col in ["bpm", "steps", "calories", "weight", "bmi", "body_fat_rate"]:
        df_weight[col] = df_weight["Value"].apply(lambda x: json.loads(x).get(col, pd.NA))

    # Convert the 'Value' column from string to dictionary
    df_sport["Value"] = df_sport["Value"].apply(json.loads)

    common_metrics = ["duration", "calories"]
    distance_metrics = ["distance"]
    ground_distance_metrics = ["min_pace", "max_pace"]
    hr_metrics = ["avg_hrm", "min_hrm", "max_hrm"]
    metrics_by_sport = {
        "basketball": [*common_metrics, *hr_metrics],
        "high_interval_training": [*common_metrics, *hr_metrics],
        "outdoor_hiking": [*common_metrics, *distance_metrics, *ground_distance_metrics],
        "outdoor_riding": [*common_metrics, *distance_metrics, *ground_distance_metrics],
        "outdoor_running": [*common_metrics, *distance_metrics, *ground_distance_metrics],
        "pool_swimming": [*common_metrics, *distance_metrics, "turn_count"],
    }

    all_cols = list(set(reduce(operator.iadd, metrics_by_sport.values(), [])))

    # Extract relevant information from the 'Value' column
    for col in all_cols:
        df_sport[col] = df_sport["Value"].apply(lambda x: x.get(col, pd.NA))
    df_weight["Time"] = pd.to_datetime(df_weight["Time"], unit="s")

    # Convert 'Time' column to datetime
    df_sport["Time"] = pd.to_datetime(df_sport["Time"], unit="s")

    return transform_df_sport(df_sport), df_weight


def plot_weight(df: pd.DataFrame, col1: str, col2: str):
    return px.scatter(
        data_frame=df[(df["Key"] == col1) & (df[col2] > 0)],
        x="Time",
        y=col2,
        color_discrete_sequence=m_c.colours_of_activities,
    )


def plot_other(df: pd.DataFrame, col1: str, col2: str):
    return px.line(
        data_frame=(
            df[(df["Key"] == col1) & (df[col2] > 0)]
            .groupby(pd.DatetimeIndex(df["Time"]).isocalendar().week.reset_index()["week"])
            .agg({col2: "mean", "Time": "first"})
            .reset_index()
        ),
        x="Time",
        y=col2,
        color_discrete_sequence=m_c.colours_of_activities,
        labels={"Time": "Time (week)"},
    )
