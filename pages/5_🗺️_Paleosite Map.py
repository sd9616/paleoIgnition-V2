import pandas as pd
import streamlit as st
import plotly.express as px

# Load the CSV file
df = pd.read_csv('charcoal_records.csv')

# Streamlit title
st.title('Map of Paleofire Sites')
st.markdown("""
            * Hover over a dot for the name of the site and its coordinates!
            """)

# Drop rows where latitude or longitude is missing
df_cleaned = df.dropna(subset=['latitude', 'longitude'])

# Create a scatter mapbox figure
fig = px.scatter_mapbox(
    df_cleaned,
    lat='latitude',
    lon='longitude',
    hover_name='site_name',
    hover_data=['latitude', 'longitude'],
    color_discrete_sequence=['red'],
    mapbox_style='open-street-map'
)


fig.update_layout(
    autosize=True,
    margin={"r":0,"t":0,"l":0,"b":0},
    height=500,# Adjust the height as needed
    mapbox=dict(
        center={"lat": df_cleaned['latitude'].mean(), "lon": df_cleaned['longitude'].mean()},
        zoom=0.5  # Adjust zoom level; a higher value means closer zoom
    )
)

# Show the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
