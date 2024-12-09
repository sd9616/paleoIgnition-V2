import streamlit as st 
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
col1, col2 = st.columns(2)

with col1: 
    st.write("# Welcome to PaleoIgnition!")
    st.markdown(
        """
        PaleoIgnition is an app built to provide paleofire practitioners with a paleolightning reconstruction for their study sites. 
        
        * Head to the __Lightning Time Series Plotter__ page on the left side bar to create a reconstruction of any site
        * Head to the __Lightning Map Plotter__ page to create heat maps for any point in time 
        * Find out more about the datasets used on the __Dataset Info + References__ page
        
        """)

with col2: 
    st.image("paleoignition_logo_white.png", width=500)

# Load the CSV file
df = pd.read_csv('charcoal_records.csv')

# Streamlit title
st.title('Paleofire Sites')

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


# df = pd.read_csv('charcoal_records.csv')



# # Streamlit map requires a dataframe with 'latitude' and 'longitude' columns
# st.title('Map of Sites')

# # Drop rows where latitude or longitude is missing
# df_cleaned = df.dropna(subset=['latitude', 'longitude'])

# # Display the cleaned map with valid latitude and longitude
# st.map(df_cleaned[['latitude', 'longitude']])

# # Optionally: Show a table with only the valid rows (with no missing values)
# st.dataframe(df_cleaned)
