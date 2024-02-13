import constants_functions as m_c
import datetime as dt

m_c.st.set_page_config(
    page_title="AthleteIQ", 
    page_icon="🚀" 
    , layout='wide'
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
        m_c.st.caption("Many fitness tracking providers enable users to download their activity data.")
        m_c.st.caption("This dashboard only needs an Excel .csv file containing a list of your activities with the specified column names. __Ensure that the column names match exactly.__")
        m_c.st.caption("__Mandatory columns__")
        m_c.st.caption("""__Activity Date__ [including timestamp], __Activity Type__ [allowed values "__Run__", "__Ride__", "__Hike__", "__Walk__", "__Swim__"], __Distance__ [in meters], __Elevation Gain__ [in meters], __Elapsed Time__ [in minutes] and __Moving Time__ [in minutes].""")
        m_c.st.caption("__Optional columns__")
        m_c.st.caption("""__Activity Name__, __Activity Description__, __Elevation Loss__, __Elevation Low__ and __Elevation High__ [all elevations in meters].""")
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
        m_c.st.warning("No data to display. Adjust filters on the side menu.")
        m_c.st.stop()
    else:
        # Create the layout
        slider_filtered_data = m_c.slider_filter('Slide to filter the starting and ending dates of activities.',filtered_df)
        slider_filtered_df = slider_filtered_data[0]
        number_of_days_on_slider = slider_filtered_data[1]
        if number_of_days_on_slider == 0:
                   m_c.st.warning("No data to display. Adjust slider to increase date range")
                   m_c.st.stop()
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
                    # 
        
        activity_types_as_list = list(slider_filtered_df['Activity Type'].unique())
        activity_selection = m_c.st.multiselect(
            "Select activities to display on charts below",
            options=activity_types_as_list,
            default = activity_types_as_list
        )

        slider_filtered_df = slider_filtered_df[slider_filtered_df['Activity Type'].isin(activity_selection)]
        data_aggregation = slider_filtered_df.copy()
        data_aggregation.set_index('Activity Date', inplace=True)
        # Create dataframe with a weekly aggregation for the area graphs
        weekly_df = data_aggregation.groupby('Activity Type').resample('W-Mon', closed='left', label='left').agg({'Elapsed Time (hours)': 'count','Moving Time (hours)': 'sum', 'Elevation Gain': 'sum', 'Distance': 'sum'})
        weekly_df.reset_index(inplace=True)
        selection_area_graph = m_c.alt.selection_point(fields=['Activity Type'], bind='legend')
        col1, col2 = m_c.st.columns(2)
        with col1:
            m_c.st.subheader("Weekly breakdown by:")            
            tab1, tab2, tab3, tab4 = m_c.st.tabs(["Moving Time", "Distance", "Number of Activites", "Elevation Gain"])
            with tab1:
                moving_time_area_graph = m_c.create_bar_chart(weekly_df, 'Moving Time (hours)', units)
                m_c.st.altair_chart(moving_time_area_graph, use_container_width=True)
                m_c.st.caption("__The accumulated moving time for each activity throughout the week (from Monday to Sunday) is summarised and visualised in a bar chart.__")
            with tab2:
                distance_area_graph = m_c.create_bar_chart(weekly_df, 'Distance', units)
                m_c.st.altair_chart(distance_area_graph, use_container_width=True)
                m_c.st.caption("__The accumulated distance for each activity throughout the week (from Monday to Sunday) is summarised and visualised in a bar chart.__")
            with tab3:
                elevation_area_graph = m_c.create_bar_chart(weekly_df, 'Elapsed Time (hours)', units)
                m_c.st.altair_chart(elevation_area_graph, use_container_width=True)
                m_c.st.caption("__The accumulated elevation gain for each activity throughout the week (from Monday to Sunday) is summarised and visualised in a bar chart.__")
            with tab4:
                elevation_area_graph = m_c.create_bar_chart(weekly_df, 'Elevation Gain', units)
                m_c.st.altair_chart(elevation_area_graph, use_container_width=True)
                m_c.st.caption("__The accumulated elevation gain for each activity throughout the week (from Monday to Sunday) is summarised and visualised in a bar chart.__")

        with col2:
            m_c.st.subheader("Relationship between:")
            tab1, tab2, tab3 = m_c.st.tabs(["Moving Time and Distance", "The Trend of Average Speed", "Weighted Average Speed and Time of Day"])
            with tab1:
                scatter_graph = m_c.create_scatter_graph(slider_filtered_df, units)
                m_c.st.altair_chart(scatter_graph, use_container_width=True)

            with tab2:
                average_speed_line_chart = m_c.create_average_speed_line_chart(slider_filtered_df, units)
                m_c.st.altair_chart(average_speed_line_chart, use_container_width=True)
                m_c.st.caption("__This is a simple calculation of average speed [activity distance divided by activity moving time] displayed as a rolling average. It is not equivalent to your pace.__")

            with tab3:
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

            m_c.st.dataframe(slider_filtered_df,
                        column_config={
                            "Activity ID": m_c.st.column_config.NumberColumn(format="%d")
                        })
    m_c.st.write("---")
    m_c.st.caption("_Users are advised that they are solely responsible for their utilisation of this dashboard. The creator of the dashboard bears no responsibility for the accuracy, interpretation or consequences of the data presented. It is imperative for users to exercise discretion and validate information independently before making any decisions based on the dashboard's content._")        
    