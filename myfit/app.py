"""Frontend for myfit analysis."""

import json
import operator
from functools import reduce
from io import StringIO

import constants_functions as m_c
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from myfit import WORKDIR


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


import zipfile

uploaded_file = "20241010_2339475430_MiFitness_ams1_data_copy.zip"


def handle_file_upload(uploaded_file, password):
    sport_filename = f"{uploaded_file.name[:29]}_hlth_center_sport_record.csv"
    fitness_filename = f"{uploaded_file.name[:29]}_hlth_center_fitness_data.csv"
    pwd = password.encode()
    zfile = zipfile.ZipFile(uploaded_file)
    df_sport = pd.read_csv(zfile.open(sport_filename, mode="r", pwd=pwd))
    df_weight = pd.read_csv(zfile.open(fitness_filename, mode="r", pwd=pwd))
    return df_sport, df_weight


@st.cache_data
def load_data(uploaded_file, password):
    # Read the data
    df_sport, df_weight = handle_file_upload(uploaded_file, password)

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


# Assuming df_sport is your original dataframe
# df_transformed = transform_df_sport(df_sport)
# print(df_transformed.dtypes)
# print(df_transformed.head())

# # # Filter only swimming activities
# # swimming_df_sport = df_sport[df_sport['Category'] == 'swimming']

# # # 1. Time Series Plot of Distance and Duration
# # fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# # fig1.add_trace(
# #     go.Scatter(x=swimming_df_sport['Time'], y=swimming_df_sport['Distance'], name="Distance"),
# #     secondary_y=False,
# # )

# # fig1.add_trace(
# #     go.Scatter(x=swimming_df_sport['Time'], y=swimming_df_sport['Duration'], name="Duration"),
# #     secondary_y=True,
# # )

# # fig1.update_layout(
# #     title_text="Swimming Distance and Duration Over Time",
# #     xaxis_title="Date",
# # )

# # fig1.update_yaxes(title_text="Distance (m)", secondary_y=False)
# # fig1.update_yaxes(title_text="Duration (s)", secondary_y=True)

# # # 2. Scatter Plot of Distance vs. Calories
# # fig2 = px.scatter(swimming_df_sport, x='Distance', y='Calories',
# #                   title='Distance vs. Calories Burned',
# #                   labels={'Distance': 'Distance (m)', 'Calories': 'Calories Burned'},
# #                   trendline='ols')

# # # 3. Box Plot of Average SWOLF Scores
# # fig3 = px.box(swimming_df_sport, y='Avg_SWOLF',
# #               title='Distribution of Average SWOLF Scores',
# #               labels={'Avg_SWOLF': 'Average SWOLF Score'})

# # # 4. Histogram of Stroke Count
# # fig4 = px.histogram(swimming_df_sport, x='Stroke_Count',
# #                     title='Distribution of Stroke Count',
# #                     labels={'Stroke_Count': 'Stroke Count'})

# # # 5. Line Plot of Performance Improvement (Distance/Duration) Over Time
# # swimming_df_sport['Performance'] = swimming_df_sport['Distance'] / swimming_df_sport['Duration']
# # fig5 = px.line(swimming_df_sport, x='Time', y='Performance',
# #                title='Swimming Performance Improvement Over Time',
# #                labels={'Time': 'Date', 'Performance': 'Distance/Duration (m/s)'})

# # # Display the figures
# # # fig1.show()
# # # fig2.show()
# # # fig3.show()
# # # fig4.show()
# # # fig5.show()

# # # Basic statistical analysis
# # print(swimming_df_sport[['Distance', 'Duration', 'Calories', 'Avg_SWOLF', 'Stroke_Count']].describe())

# # # Correlation analysis
# # correlation_matrix = swimming_df_sport[['Distance', 'Duration', 'Calories', 'Avg_SWOLF', 'Stroke_Count']].corr()
# # print("\nCorrelation Matrix:")
# # print(correlation_matrix)


# # 1. Activity Distribution
# activity_counts = df_sport["Category"].value_counts()
# fig1 = px.pie(
#     values=activity_counts.values, names=activity_counts.index, title="Distribution of Activities"
# )

# # 2. Time Series Plot of Activities
# fig2 = px.scatter(
#     df_sport,
#     x="Time",
#     y="Duration",
#     color="Category",
#     title="Activities Over Time",
#     labels={"Time": "Date", "Duration": "Duration (s)"},
#     hover_data=["Distance", "Calories"],
# )

# # 3. Box Plot of Calories Burned by Activity
# fig3 = px.box(
#     df_sport,
#     x="Category",
#     y="Calories",
#     title="Calories Burned by Activity Type",
#     labels={"Calories": "Calories Burned"},
# )

# # 4. Scatter Plot of Duration vs. Calories for All Activities
# fig4 = px.scatter(
#     df_sport,
#     x="Duration",
#     y="Calories",
#     color="Category",
#     title="Duration vs. Calories Burned (All Activities)",
#     labels={"Duration": "Duration (s)", "Calories": "Calories Burned"},
#     trendline="ols",
# )

# # 5. Line Plot of Distance Over Time for Running and Swimming
# running_swimming = df_sport[df_sport["Category"].isin(["running", "swimming"])]
# fig5 = px.line(
#     running_swimming,
#     x="Time",
#     y="Distance",
#     color="Category",
#     title="Distance Over Time (Running vs Swimming)",
#     labels={"Time": "Date", "Distance": "Distance (m)"},
# )

# # 6. Histogram of Average Heart Rate by Activity
# fig6 = px.histogram(
#     df_sport[df_sport["Avg_HR"] > 0],
#     x="Avg_HR",
#     color="Category",
#     title="Distribution of Average Heart Rate by Activity",
#     labels={"Avg_HR": "Average Heart Rate"},
#     marginal="box",
# )

# # 7. Scatter Plot of Steps vs. Calories for Walking and Running
# walking_running = df_sport[df_sport["Category"].isin(["walking", "running"])]
# fig7 = px.scatter(
#     walking_running,
#     x="Steps",
#     y="Calories",
#     color="Category",
#     title="Steps vs. Calories (Walking and Running)",
#     labels={"Steps": "Step Count", "Calories": "Calories Burned"},
#     trendline="ols",
# )

# # 8. Heatmap of Correlation Matrix
# numeric_cols = ["Duration", "Distance", "Calories", "Avg_HR", "Max_HR", "Steps"]
# correlation_matrix = df_sport[numeric_cols].corr()
# fig8 = px.imshow(correlation_matrix, title="Correlation Matrix of Fitness Metrics")

# # Display the figures
# # fig1.show()
# # fig2.show()
# # fig3.show()
# # fig4.show()
# # fig5.show()
# # fig6.show()
# # fig7.show()
# # fig8.show()

# # Basic statistical analysis by activity
# for category in df_sport["Category"].unique():
#     print(f"\nStatistics for {category}:")
#     print(df_sport[df_sport["Category"] == category][numeric_cols].describe())

# # Overall correlation analysis
# print("\nOverall Correlation Matrix:")
# print(correlation_matrix)


# import pandas as pd


# def plot_activity_metrics(df_sport, activity):
#     # Filter data for the specific activity
#     activity_df_sport = df_sport[df_sport["Category"] == activity]

#     # Create subplots
#     fig = make_subplots(
#         rows=4,
#         cols=1,
#         subplot_titles=(
#             "Distance over Time",
#             "Calories Burned over Time",
#             "Heart Rate over Time",
#             "Combined Metrics over Time",
#         ),
#         vertical_spacing=0.1,
#         shared_xaxes=True,
#     )

#     # Plot Distance over Time
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Distance"],
#             mode="lines+markers",
#             name="Distance",
#         ),
#         row=1,
#         col=1,
#     )

#     # Plot Calories over Time
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Calories"],
#             mode="lines+markers",
#             name="Calories",
#         ),
#         row=2,
#         col=1,
#     )

#     # Plot Heart Rate over Time
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Avg_HR"],
#             mode="lines+markers",
#             name="Avg HR",
#         ),
#         row=3,
#         col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Max_HR"],
#             mode="lines+markers",
#             name="Max HR",
#         ),
#         row=3,
#         col=1,
#     )

#     # Combined plot
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Distance"],
#             mode="lines+markers",
#             name="Distance",
#             yaxis="y4",
#         ),
#         row=4,
#         col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Calories"],
#             mode="lines+markers",
#             name="Calories",
#             yaxis="y5",
#         ),
#         row=4,
#         col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df_sport["Time"],
#             y=activity_df_sport["Avg_HR"],
#             mode="lines+markers",
#             name="Avg HR",
#             yaxis="y6",
#         ),
#         row=4,
#         col=1,
#     )

#     # Update layout
#     fig.update_layout(height=1200, width=800, title_text=f"{activity} Metrics Over Time")
#     fig.update_xaxes(title_text="Date", row=4, col=1)
#     fig.update_yaxes(title_text="Distance", row=1, col=1)
#     fig.update_yaxes(title_text="Calories", row=2, col=1)
#     fig.update_yaxes(title_text="Heart Rate", row=3, col=1)

#     # Set up multiple y-axes for the combined plot
#     fig.update_layout(
#         yaxis4=dict(title="Distance", side="left", position=0.05),
#         yaxis5=dict(title="Calories", side="left", position=0.15, overlaying="y4", anchor="x"),
#         yaxis6=dict(title="Heart Rate", side="right", overlaying="y4", anchor="x"),
#     )

#     fig.show()


# # Get unique activities
# activities = df_sport["Category"].unique()

# # Generate plots for each activity
# # for activity in activities:
# #     plot_activity_metrics(df_sport, activity)


# df_weight["Time"] = pd.to_datetime(df_weight["Time"], unit="s")
# df_weight["weight"] = df_weight["Value"].apply(lambda x: json.loads(x).get("weight", 0))
# df_weight["bmi"] = df_weight["Value"].apply(lambda x: json.loads(x).get("bmi", 0))
# df_weight["body_fat_rate"] = df_weight["Value"].apply(
#     lambda x: json.loads(x).get("body_fat_rate", 0)
# )

# px.scatter(data_frame=df_weight, x="Time", y="weight")
# px.scatter(data_frame=df_weight, x="Time", y="bmi")
# px.scatter(data_frame=df_weight[df_weight["body_fat_rate"] > 0], x="Time", y="body_fat_rate")


# import streamlit as st

# st.set_page_config(layout="wide")

# with st.sidebar:
#     st.markdown("### Empty sidebar")

# container1 = st.container()
# col1, col2 = st.columns(2)

# with container1:
#     with col1:
#         st.plotly_chart(fig1, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig2, use_container_width=True)


# container2 = st.container()
# col3, col4 = st.columns(2)

# with container2:
#     with col3:
#         fig3
#     with col4:
#         fig4
