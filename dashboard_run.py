import constants_functions as m_c
import datetime as dt

m_c.st.set_page_config(
    page_title="AthleteIQ", 
    page_icon="🏆" 
    , layout="wide"
    )

if __name__ == '__main__': 
    with m_c.st.container():
        m_c.st.header("🅰🆃🅷🅻🅴🆃🅴 🅸🆀")
        m_c.st.subheader('''An Interactive View of your Fitness Data''')
    with m_c.st.sidebar:
        m_c.st.write("🌐 [Contact](https://www.linkedin.com/in/matthew-helingoe-55371791/)  |  🚧 [GitHub](https://github.com/shoulda-woulda/fitness_dashboard_app)")
        m_c.st.caption("_This dashboard is agnostic of specific fitness tracking providers and exclusively retains your data in a cache that is cleared at the conclusion of your session._")
        m_c.st.caption("_Optimal performance is achieved on a desktop platform._")
        m_c.st.write("---")
        m_c.st.header("How to upload your data")
        m_c.st.caption("Many fitness tracking providers and devices enable users to download their activity data.")
        m_c.st.caption("This dashboard only needs an Excel .csv file containing a list of your activities with the specified column names. __Ensure that the column names match exactly.__")
        m_c.st.caption("__Mandatory columns__")
        m_c.st.caption("""__Activity Date__ [including timestamp], __Activity Type__ [allowed values "_Run_", "_Ride_", "_Hike_", "_Walk_", "_Swim_"], __Distance__ [in meters], __Elevation Gain__ [in meters], __Elapsed Time__ [in minutes] and __Moving Time__ [in minutes].""")
        m_c.st.caption("__Optional columns__")
        m_c.st.caption("""__Activity ID__, __Activity Name__, __Activity Description__, __Elevation Loss__, __Elevation Low__ and __Elevation High__ [all elevations in meters].""")
        m_c.st.caption("""Any additional columns present in your file will be automatically removed.""")
        m_c.st.caption(" Select your desired unit of measurement (don\'t worry you can change it later), drop the file below and enjoy!")
        units = m_c.st.radio("Select units", [('km', 'm', 'kph'), ('mi', 'ft', 'mph')])
        uploaded_file = m_c.st.sidebar.file_uploader("""Upload your activities .csv file""", type=["csv"])
    if uploaded_file is None:
        m_c.st.info("⬅️ Get started and upload your own data on the side menu.")
        m_c.st.write("---")
        file_ = open("misc/dashboard-demo.gif", "rb")
        contents = file_.read()
        data_url = m_c.base64.b64encode(contents).decode("utf-8")
        file_.close()
        m_c.st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        m_c.st.stop()
    else:
            df = m_c.load_data(uploaded_file,units)
            with m_c.st.sidebar:
                filtered_df = m_c.filter_dataframe(df, units)
                m_c.st.header("How to use the dashboard")
                m_c.st.caption("You can filter your activity data above or use the options on the dashboard.")
                m_c.st.caption("The large figures represent a breakdown of your total activity metrics based on the applied filters.")
                m_c.st.caption("Use the dropdown to filter the graphs by activity type.")
                m_c.st.caption("__Desktop only__ The graphs are interactive, allowing you to zoom in and out. Hover over data points for additional details. Double-click to reset the graph to its previous state.")
                m_c.st.caption("You can view and download your data in the table at the bottom of the page.")
    if filtered_df is None:
        m_c.st.stop()
    elif filtered_df.empty:
        m_c.st.error("No data to display. Ensure your csv file isn't blank and adjust filters on the side menu.", icon="🚨")
        # m_c.st.warning("")
        m_c.st.stop()
    else:
        # Create the layout
        slider_filtered_data = m_c.slider_filter('Slide to filter the starting and ending dates of activities.',filtered_df)
        slider_filtered_df = slider_filtered_data[0]
        number_of_days_on_slider = slider_filtered_data[1]
        min_date_on_slider = slider_filtered_data[2]
        cutoff_date = min_date_on_slider - dt.timedelta(days=number_of_days_on_slider)
        filtered_df_to_create_perc_difference = filtered_df[(filtered_df['Activity Date'].dt.date >= cutoff_date) & (filtered_df['Activity Date'].dt.date < min_date_on_slider)]    
        prev_totals_df = filtered_df_to_create_perc_difference.groupby('Activity Type').agg({'Activity Date': 'count', 'Moving Time (hours)': 'sum', 'Elevation Gain': 'sum', 'Distance': 'sum'})
        prev_totals_df.reset_index(inplace=True)
        totals_df = slider_filtered_df.groupby('Activity Type').agg({'Activity Date': 'count', 'Moving Time (hours)': 'sum', 'Elevation Gain': 'sum', 'Distance': 'sum'})
        totals_df.reset_index(inplace=True)
        # Merge dataframes on 'Activity Type'
        merged_df = m_c.pd.merge(prev_totals_df, totals_df, on='Activity Type', how='outer', suffixes=('_df1', '_df2'))
        # Calculate % difference for each metric
        merged_df['Activity Date'] = (merged_df['Activity Date_df2'] - merged_df['Activity Date_df1']) / merged_df['Activity Date_df1'] * 100
        merged_df['Moving Time (hours)'] = (merged_df['Moving Time (hours)_df2'] - merged_df['Moving Time (hours)_df1']) / merged_df['Moving Time (hours)_df1'] * 100
        merged_df['Elevation Gain'] = (merged_df['Elevation Gain_df2'] - merged_df['Elevation Gain_df1']) / merged_df['Elevation Gain_df1'] * 100
        merged_df['Distance'] = (merged_df['Distance_df2'] - merged_df['Distance_df1']) / merged_df['Distance_df1'] * 100
        merged_df['Activity Date'] = round(merged_df['Activity Date'],2)
        merged_df['Moving Time (hours)'] = round(merged_df['Moving Time (hours)'],2)
        merged_df['Elevation Gain'] = round(merged_df['Elevation Gain'],2)
        merged_df['Distance'] = round(merged_df['Distance'],2)
        totals_df['Activity Date'] = round(totals_df['Activity Date'],2)
        totals_df['Moving Time (hours)'] = round(totals_df['Moving Time (hours)'],2)
        totals_df['Elevation Gain'] = round(totals_df['Elevation Gain'],2)
        totals_df['Distance'] = round(totals_df['Distance'],2)
        totals_df['Moving Time (hours)'] = totals_df['Moving Time (hours)'].apply(m_c.format_hours)
        # Convert Activity Type column to categorical so it can be sorted
        totals_df['Activity Type'] = m_c.pd.Categorical(totals_df['Activity Type'], categories=m_c.allowable_activities, ordered=True)
        merged_df['Activity Type'] = m_c.pd.Categorical(merged_df['Activity Type'], categories=m_c.allowable_activities, ordered=True)
        # Actually sort Activity Type column
        totals_df = totals_df.sort_values(by='Activity Type')
        merged_df = merged_df.sort_values(by='Activity Type')
        # Enumerate the unique values in Activity Type column and create metric elements for each activity type
        caption_text = "⬆️⬇️ Percentage changes calculated between the selected date range and an equivalent duration preceding [" + str(number_of_days_on_slider) + " days between  " + str(cutoff_date) + " & " +  str(min_date_on_slider) + "]. A nan value means no previous data for that activity type."
        m_c.st.info(caption_text)
        caption_text = "Totals and Percentage Changes"
        m_c.st.subheader(caption_text)
        for i, activity_type in enumerate(list(totals_df['Activity Type'].unique()), start=1):
            with m_c.st.container():
                m_c.st.subheader(f"{activity_type}")
                if i == 1 or i == 3:
                    m_c.create_metrics(totals_df, merged_df, activity_type, (i*i), (i*i)+1, (i*i)+2, (i*i)+3, units)
                if i == 2 or i == 4:
                    m_c.create_metrics(totals_df, merged_df, activity_type, (2*i)+1, (2*i)+2, (2*i)+3, (2*i)+4, units)
        activity_types_as_list = list(slider_filtered_df['Activity Type'].unique())
        activity_selection = m_c.st.multiselect(
            "Select activity types to display on charts below:",
            options=activity_types_as_list,
            default = activity_types_as_list
        )
        slider_filtered_df = slider_filtered_df[slider_filtered_df['Activity Type'].isin(activity_selection)]
        if slider_filtered_df.empty:
            m_c.st.error("No data to display. Adjust filters.", icon="🚨")
            m_c.st.stop()
        data_aggregation = slider_filtered_df.copy()
        data_aggregation.set_index('Activity Date', inplace=True)
        # Create dataframe with a weekly aggregation for the area graphs
        weekly_df = data_aggregation.groupby('Activity Type').resample('W-Mon', closed='left', label='left').agg({'Elapsed Time (hours)': 'count','Moving Time (hours)': 'sum', 'Elevation Gain': 'sum', 'Distance': 'sum'})
        weekly_df.reset_index(inplace=True)
        selection_area_graph = m_c.alt.selection_point(fields=['Activity Type'], bind='legend')
        tab1, tab2 = m_c.st.tabs(["Historic and Current View of Activities", "Future Estimation of Activities"])
        with tab1:
            col1, col2 = m_c.st.columns(2)
            with col1:
                m_c.st.subheader("Weekly breakdown by:")            
                tab3, tab4, tab5, tab6= m_c.st.tabs(["Moving Time", "Distance", "Number of Activites", "Elevation Gain"])
                with tab3:
                    moving_time_area_graph = m_c.create_area_chart(weekly_df, 'Moving Time (hours)', units)
                    m_c.st.altair_chart(moving_time_area_graph, use_container_width=True)
                    m_c.st.caption("The accumulated moving time for each activity type throughout the week (from Monday to Sunday) is summarised and visualised in an area chart.")
                with tab4:
                    distance_area_graph = m_c.create_area_chart(weekly_df, 'Distance', units)
                    m_c.st.altair_chart(distance_area_graph, use_container_width=True)
                    m_c.st.caption("The accumulated distance for each activity type throughout the week (from Monday to Sunday) is summarised and visualised in an area chart.")
                with tab5:
                    no_of_activities_area_graph = m_c.create_area_chart(weekly_df, 'Elapsed Time (hours)', units)
                    m_c.st.altair_chart(no_of_activities_area_graph, use_container_width=True)
                    m_c.st.caption("The number of activities for each activity type throughout the week (from Monday to Sunday) is summarised and visualised in an area chart.")
                with tab6:
                    elevation_area_graph = m_c.create_area_chart(weekly_df, 'Elevation Gain', units)
                    m_c.st.altair_chart(elevation_area_graph, use_container_width=True)
                    m_c.st.caption("The accumulated elevation gain for each activity type throughout the week (from Monday to Sunday) is summarised and visualised in an area chart.")
            with col2:
                m_c.st.subheader("Relationship between:")
                tab7, tab8, tab9 = m_c.st.tabs(["Moving Time and Distance", "The Trend of Average Speed", "Weighted Average Speed and Time of Day"])
                with tab7:
                    scatter_graph = m_c.create_scatter_graph(slider_filtered_df, units)
                    m_c.st.altair_chart(scatter_graph, use_container_width=True)
                with tab8:
                    average_speed_line_chart = m_c.create_average_speed_line_chart(slider_filtered_df, units)
                    m_c.st.altair_chart(average_speed_line_chart, use_container_width=True)
                    m_c.st.caption("This is a simple calculation of average speed [activity distance divided by activity moving time] displayed as a rolling average. It is not equivalent to your pace.")
                with tab9:
                    slider_filtered_df['Hour'] = slider_filtered_df['Activity Date'].dt.hour
                    slider_filtered_df['Period of Day'] = slider_filtered_df['Hour'].apply(m_c.categorise_period)
                    # Remove any instances where there is an inf value in either the speed or distance (can occur when tracking activity on a treadmill).
                    filtered_df = slider_filtered_df[~slider_filtered_df[['Average Speed', 'Distance']].isin([m_c.np.inf, -m_c.np.inf]).any(axis=1)]
                    weighted_avg_speed = filtered_df.groupby(['Activity Type', 'Period of Day']).apply(lambda x: (x['Distance'] * x['Average Speed']).sum() / x['Distance'].sum()).reset_index(name='Weighted Avg Speed')
                    chart_weighted_average = m_c.create_mark_bar_weighted_average(weighted_avg_speed, units)
                    m_c.st.altair_chart(chart_weighted_average, use_container_width=True)
                    m_c.st.caption("Definitions: __Morning__: 5am - 11am, __Afternoon__: 11am - 5pm, __Evening__: 5pm - 10pm, __Night__: 10pm - 5am")
                    m_c.st.caption("The weighted average speed for each activity is computed by multiplying the distance and speed of each activity, summing these products, and then dividing by the total distance across all activities. This metric offers a representation of the overall distance covered. Mathematically this is represented by:")
                    m_c.st.latex(r'''\text{Weighted Average Speed} = \frac{\sum \text{Distance} \times \text{Speed}}{\sum \text{Distance}}​''')
                    m_c.st.caption("The weighted average speed takes into account both the distance covered and the speed achieved during activities. This provides a more comprehensive measure compared to looking at distance or speed in isolation.")
                    m_c.st.caption("The breakdown is included to offer insights into your performance levels during different times of the day, helping identify when you perform at your best.")
        with tab2:
                data_for_regression_tab = data_aggregation.copy()
                data_for_regression_tab.reset_index(inplace=True)
                m_c.st.subheader("Configure the Prediction Model")
                col1, col2, col3 = m_c.st.columns(3)
                today = m_c.dt.datetime.today()
                max_date_of_activities = max(data_for_regression_tab['Activity Date'].dt.date)
                min_date_of_activities = min(data_for_regression_tab['Activity Date'].dt.date)
                # The following code creates the dates for the predicition model.
                with col1:
                    start_date = m_c.st.date_input("To calculate my predicted values use activity data beginning from ", value=m_c.dt.date(2024, 1, 1))
                    m_c.st.info("Select a date when you began consistently training. At least 2 days of activity data is required to calculate a prediction.")
                    time_difference = max_date_of_activities - start_date
                    number_of_days_on_slider = time_difference.days + 1
                with col2:
                    number = m_c.st.number_input("AND calculate my predictions until ... many days afterwards", step=1, value=None, placeholder="...", min_value=number_of_days_on_slider)
                    if number == None:
                        end_date = m_c.st.date_input("or until this date ", value=today + dt.timedelta(days=90),min_value=max_date_of_activities)
                    else:
                        end_date = m_c.st.date_input("or until this date ", value=start_date + dt.timedelta(days=number),min_value=start_date+ dt.timedelta(days=1))
                with col3:
                    number_of_days_between_end_date_and_max_date_of_activities = (end_date-max_date_of_activities).days
                    # Predictor 1 is by default set at halfway between the end date of the regression line and the max date of activities
                    predictor_date_1 = m_c.st.date_input("AND I want to predict my activity values on ", value=end_date- dt.timedelta(days=number_of_days_between_end_date_and_max_date_of_activities*0.8), max_value=end_date)
                    # Predictor 2 is by default set at 80% between the end date of the regression line and the max date of activities
                    predictor_date_2 = m_c.st.date_input("and on ", value=end_date - dt.timedelta(days=round(number_of_days_between_end_date_and_max_date_of_activities*0.2)), max_value=end_date)
                # Filter the data on the chosen start and end dates using a mask
                mask = data_for_regression_tab['Activity Date'].dt.date.between(start_date,end_date)
                slider_filtered_df_regression = data_for_regression_tab[mask]
                slider_filtered_df_regression.reset_index(inplace=True)
                # Create cumulative columns for distance, moving time, elevation gain
                slider_filtered_df_regression["Cumulative Distance"] = slider_filtered_df_regression.groupby(["Activity Type"])["Distance"].cumsum()
                slider_filtered_df_regression["Cumulative Moving Time (hours)"] = slider_filtered_df_regression.groupby(["Activity Type"])["Moving Time (hours)"].cumsum()
                slider_filtered_df_regression["Cumulative Elevation Gain"] = slider_filtered_df_regression.groupby(["Activity Type"])["Elevation Gain"].cumsum()
                max_cumulative_distance = slider_filtered_df_regression.groupby('Activity Type')['Cumulative Distance'].max()
                # Then sort the DataFrame based on activity type and the maximum cumulative distance
                slider_filtered_df_regression = slider_filtered_df_regression.sort_values(by=['Activity Type', 'Cumulative Distance'], ascending=[True, False])
                # Finally, reorder the DataFrame based on the maximum cumulative distance for each activity type
                slider_filtered_df_regression['Activity Type'] = m_c.pd.Categorical(slider_filtered_df_regression['Activity Type'], categories=max_cumulative_distance.index, ordered=True)
                grouped_df = slider_filtered_df_regression.groupby('Activity Type')
                # Filter out groups with only one row
                slider_filtered_df_regression = grouped_df.filter(lambda x: len(x) > 1)
                slider_filtered_df_regression.reset_index(inplace=True)
                slider_filtered_df_regression = slider_filtered_df_regression.sort_values(by='Activity Date')
                tab10, tab11, tab12 = m_c.st.tabs(["Distance", "Moving Time", "Elevation Gain"])
                with tab10:
                    combined_chart, numbers_of_activities, r_squared_values, mae_values, predicted_values_activity_type = m_c.create_scatter_graph_and_regression(slider_filtered_df_regression,'Cumulative Distance', units, predictor_date_1, predictor_date_2, end_date)
                    col1, col2 = m_c.st.columns(2)
                    with col1:
                        m_c.st.altair_chart(combined_chart, use_container_width=True)
                    with col2:
                        m_c.create_columns_for_statistics(slider_filtered_df_regression, predicted_values_activity_type, r_squared_values, mae_values, numbers_of_activities, units, 'Cumulative Distance')
                with tab11:
                    combined_chart, numbers_of_activities, r_squared_values, mae_values, predicted_values_activity_type = m_c.create_scatter_graph_and_regression(slider_filtered_df_regression,'Cumulative Moving Time (hours)', units, predictor_date_1, predictor_date_2, end_date)
                    col1, col2 = m_c.st.columns(2)
                    with col1:
                        m_c.st.altair_chart(combined_chart, use_container_width=True)
                    with col2:
                        m_c.create_columns_for_statistics(slider_filtered_df_regression, predicted_values_activity_type, r_squared_values, mae_values, numbers_of_activities, units, 'Cumulative Moving Time (hours)')
                with tab12:
                    slider_filtered_df_regression_without_swim = slider_filtered_df_regression[slider_filtered_df_regression['Activity Type'] != 'Swim']
                    combined_chart, numbers_of_activities, r_squared_values, mae_values, predicted_values_activity_type = m_c.create_scatter_graph_and_regression(slider_filtered_df_regression_without_swim, 'Cumulative Elevation Gain', units, predictor_date_1, predictor_date_2, end_date)
                    col1, col2 = m_c.st.columns(2)
                    with col1:
                        m_c.st.altair_chart(combined_chart, use_container_width=True)
                    with col2:
                        m_c.create_columns_for_statistics(slider_filtered_df_regression_without_swim, predicted_values_activity_type, r_squared_values, mae_values, numbers_of_activities, units, 'Cumulative Elevation Gain')              
                m_c.st.caption("This model __assumes a linear relationship between your completed activities and the future.__ If you changed your training volume __suddenly__, this model will lose accuracy. Adjust the configuration of the model to use activity data beginning from when you changed your training volume. Similarly, __you want to have a significant number of activities in the model calculation such that predicitions have a reliable basis__ to be found upon. Insufficient data can lead to unreliable and potentially misleading predictions.")                                 
                m_c.st.write("---")
    with m_c.st.expander("Your activity data as a table"):
            # Drop the hour column for display.
            slider_filtered_df = slider_filtered_df.drop('Hour', axis=1)
            # Adding the unit suffixes back in to the column names for display.
            elevation_suffix = f' ({units[1]})'
            distance_suffix = f' ({units[0]})'
            speed_suffix = f' ({units[2]})'
            slider_filtered_df.rename(columns={col: col + elevation_suffix for col in m_c.elevation_columns}, inplace=True)
            slider_filtered_df.rename(columns={col: col + distance_suffix for col in m_c.distance_column}, inplace=True)
            slider_filtered_df.rename(columns={col: col + speed_suffix for col in m_c.speed_column}, inplace=True)
            # Sort by Activity Date and reset index
            slider_filtered_df = slider_filtered_df.sort_values(by='Activity Date').reset_index(drop=True)
            m_c.st.dataframe(slider_filtered_df,
                        column_config={
                            "Activity ID": m_c.st.column_config.NumberColumn(format="%d")
                        })
    m_c.st.write("---")
    m_c.st.caption("_Users are advised that they are solely responsible for their utilisation of this dashboard. The creator of the dashboard bears no responsibility for the accuracy, interpretation or consequences of the data presented. It is imperative for users to exercise discretion and validate information independently before making any decisions based on the dashboard's content._")        
    