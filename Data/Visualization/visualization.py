from tokenize import group

from .visual_helper import *
import pandas as pd
import osmnx as ox
import folium
import json
import geopandas as gpd
from geopy.geocoders import Nominatim



def plot_humanitarian_needs(country_data):
    '''
    This function will plot the trend of humanitarian needs for a given country, takes HapiClass object as input
    :param country_data: a HapiClass object
    :return: None
    '''

    # In case the object has not obtained humanitarian data
    if not country_data.humanitarian_data:
        country_data.get_humanitarian_needs_data()

    df = country_data.humanitarian_data
    intersector_df = df[df['sector_name'] == 'Intersectoral']

    # Find population in need
    intersector_df = intersector_df[intersector_df['population_status'] == 'INN']

    # Disregard Whether refugee or not
    intersector_df = intersector_df[intersector_df['population_group'] == 'all']

    # Disregard Age
    intersector_df = intersector_df[intersector_df['age_range'] == 'ALL']

    # Disregard Disabled
    intersector_df = intersector_df[intersector_df['disabled_marker'] == 'all']

    # Visualization currently unavailable (Not time series data, only 2024)
    pass


def plot_conflict_events(country_data, event_type='all'):
    """
    This function will plot the conflict events for a given country, takes HapiClass object as input
    :param country_data: a HapiClass object
    :return: None
    """
    # Retrieve Data
    if not country_data.conflict_event_data:
        country_data.get_conflict_event_data()
    df = country_data.conflict_event_data

    # Add a column of year
    df['reference_period_start'] = pd.to_datetime(df['reference_period_start'])
    df['year'] = df['reference_period_start'].dt.year

    latest_year = df['year'].max()
    df = df[df['year'] >= latest_year - 10]

    # Plot
    if event_type == 'all':
        # Create bar and line plot
        events_per_year = df.groupby('year')['events'].sum().reset_index()
        x = events_per_year['year']
        y = events_per_year['events']
        line_bar_plot(x, y, title='Conflict Events Count Trend', y_label='Event Count')
        casualties_per_year = df.groupby('year')['fatalities'].sum().reset_index()
        x = casualties_per_year['year']
        y = casualties_per_year['fatalities']
        plot = line_bar_plot(x, y, title='Conflict Events Fatalities Trend', y_label='Fatalities Count')
        return plot
    return None


def plot_funding(country_data):
    """
    Plots the funding trend
    """
    if not country_data.funding_data:
        country_data.get_funding_data()
    df = country_data.funding_data

    # Add a column of year
    df['reference_period_start'] = pd.to_datetime(df['reference_period_start'])
    df['year'] = df['reference_period_start'].dt.year

    # Plot a bar and line plot, where x-axis is year, and y-axis is funding received
    funding_per_year = df.groupby('year')['funding_usd'].sum().reset_index()
    x = funding_per_year['year']
    y = funding_per_year['funding_usd']
    plot = line_bar_plot(x, y, title='Funding Trend', y_label='Amount (billion USD)', unit='billion',
                                       save_path='./funding')
    return plot


def plot_population(country_data):
    if not country_data.population_data:
        country_data.get_population_data()
    df = country_data.population_data

    df = df[df['age_range'] != 'all']

    def merge_age_range(age):
        try:
            if age == '80+':
                return '60+'
            start_age = int(age.split('-')[0])
            if start_age >= 60:
                return '60+'
            else:
                return age
        except ValueError:
            return age

    df.loc[:, 'age_range'] = df['age_range'].apply(merge_age_range)
    # print(df.columns)

    aggregated_data = df.groupby('age_range', as_index=False)['population'].sum()
    # print(aggregated_data)

    data = aggregated_data['population'].values
    labels = aggregated_data['age_range'].values

    plot = pie_chart(data, labels, title='Population age range in AFG', save_path='./population')
    return plot


def plot_events(country_data):
    """
    Process data for bar chart.

    :param country_data: DataFrame containing the data
    :return: processed x and y data for plotting
    """
    # Optional data filtering based on condition
    if not country_data.conflict_event_data:
        country_data.get_conflict_event_data()
    df = country_data.conflict_event_data

    df['reference_period_start'] = pd.to_datetime(df['reference_period_start'])
    df['year'] = df['reference_period_start'].dt.year

    # Group by 'year' and 'admin1_name' and sum the 'events'
    aggregated_data = df.groupby(['year', 'admin1_name'])['events'].sum().unstack(fill_value=0)

    # Call bar_chart for visualization
    plot = bar_chart(aggregated_data,
                                   title=f"Conflict Events Over Years in {country_data.country_name}",
                                   x_label="Year",
                                   y_label="Number of Events",
                                   save_path='./conflict_events2',
                                   color='skyblue',
                                   alpha=0.7,
                                   stacked=False)

    return plot


def plot_refugee_data(country_data):
    if not country_data.refugee_data:
        country_data.get_refugee_data()
    refugee_data = country_data.refugee_data

    # Convert reference_period_start to datetime
    refugee_data['reference_period_start'] = pd.to_datetime(refugee_data['reference_period_start'], errors='coerce')

    # Filter refugees based on the location
    # refugees = refugee_data[refugee_data['origin_location_code'] == country_data.LOCATION]
    refugee_data.to_csv('refugee_data.csv')

    # Extract the year from reference_period_start
    refugee_data['Year'] = refugee_data['reference_period_start'].dt.year

    # Group by Year and sum the population for each year
    yearly_trends = refugee_data.groupby(['Year'])['population'].sum().reset_index()

    plot = line_bar_plot(
        x=yearly_trends['Year'],
        y=yearly_trends['population'],
        title=f"Yearly Refugee Population Trends from {country_data.LOCATION}",
        x_label="Year",
        y_label="Total Refugee Population",
        save_path='./refugee'
    )

    return plot


def plot_humanitarian_needs_geo_plot(country_data):
    '''
    This function will plot the trend of humanitarian needs for a given country, taking a HapiClass object as input.
    It also generates a GeoJSON file and a choropleth map based on the humanitarian data.

    :param country_data: a HapiClass object
    :return: None
    '''

    # Ensure the object has obtained humanitarian data
    if not country_data.humanitarian_data:
        country_data.get_humanitarian_needs_data()

    df = country_data.humanitarian_data

    # Filter for Intersectoral data and population in need
    intersector_df = df[(df['sector_name'] == 'Intersectoral') & (df['population_status'] == 'all')]
    intersector_df.to_csv('data.csv')

    # Group data by admin1_name and admin2_name
    grouped_df = intersector_df.groupby(['admin1_name'])['population'].sum().reset_index(name='population')

    # Function to get geometry for a region
    def get_region_geometry(admin1_name):
        try:
            query = f"{admin1_name}, {country_data.country_name}"
            gdf = ox.geocode_to_gdf(query)
            if not gdf.empty:
                return gdf['geometry'].iloc[0]
        except Exception as e:
            print(f"Failed to get geometry for {admin1_name}: {e}")
        return None

    # Store results in a list instead of inefficiently concatenating GeoDataFrames
    results = []

    # Iterate through grouped_df to get geometry for each region
    for _, row in grouped_df.iterrows():
        admin1_name, count = row['admin1_name'], row['population']
        geometry = get_region_geometry(admin1_name)

        if geometry is not None:
            results.append(
                {'admin1_name': admin1_name, 'count': count, 'geometry': geometry})

    # Convert list to GeoDataFrame
    if results:
        results_gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")
    else:
        print("No valid geographic data found. Exiting function.")
        return

    # Convert GeoDataFrame to GeoJSON
    geojson_data = json.loads(results_gdf.to_json())

    # Save as GeoJSON file
    with open("regions.geojson", "w") as f:
        json.dump(geojson_data, f, indent=4)

    print("✅ GeoJSON file generated: regions.geojson")

    # Define the map center (default to Kabul's coordinates)
    geolocator = Nominatim(user_agent="geo_center_finder")
    location = geolocator.geocode(country_data.country_name)
    if location:
        map_center = [location.latitude, location.longitude]  # [latitude, longitude]
    else:
        map_center = [0, 0]

    # Create the map
    m = folium.Map(location=map_center, zoom_start=6)

    # Generate Choropleth map
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        data=grouped_df,
        name="Districts Statistics",
        columns=["admin1_name", "population"],
        key_on="feature.properties.admin1_name",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.8,
        legend_name="Population in Need"
    )
    choropleth.add_to(m)

    # Save the map
    m.save("choropleth_map.html")
    print("✅ Map generated, please open 'choropleth_map.html' to view")