import base64
from io import BytesIO
import os
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import pandas as pd
import plotly.express as px

TRACE = "TRACE"
FAMOUS = "FAMOUS"
LOVECLIM = "LOVECLIM"

st.markdown("""
            ## Lightning Map Plotter
            
            * Generate a heat map for any time from -21kaBP to present!
            """)


chosen_dataset = st.selectbox(
    "Pick a Dataset to plot!",
    ("TraCE", "FAMOUS", "LOVECLIM"))

# Enter lat and Lon
time_entered = st.number_input('Enter a point in time', -22.0, 0.0, step=1e-2, format="%.5f")

generate = st.button("Generate", type="primary")

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(project_path)

file_path = os.path.join(project_path, 'data')

def load_datasets(): 
    
    loveclim = 'loveclim'
    trace = 'trace'
    famous = 'famous'
    # [variable, latitude, longitude, time]
    dataset_files_loveclm = [
        np.load(os.path.join(file_path, f'{loveclim}_lightning.npy')),
        np.load(os.path.join(file_path, f'{loveclim}_lat.npy')),
        np.load(os.path.join(file_path, f'{loveclim}_lon.npy')),
        np.load(os.path.join(file_path, f'{loveclim}_time_kaBP.npy'))
        
    ]

    dataset_files_trace = [
        np.load(os.path.join(file_path, f'{trace}_lightning.npy')), 
        np.load(os.path.join(file_path, f'{trace}_lat.npy')),
        np.load(os.path.join(file_path, f'{trace}_lon.npy')),
        np.load(os.path.join(file_path, f'{trace}_time_kaBP.npy'))
        
    ]

    dataset_files_famous = [
        np.load(os.path.join(file_path, f'{famous}_lightning.npy')), 
        np.load(os.path.join(file_path, f'{famous}_lat.npy')),
        np.load(os.path.join(file_path, f'{famous}_lon.npy')),
        np.load(os.path.join(file_path, f'{famous}_time_kaBP.npy'))
        
    ]



    dataset_files_list = {
        TRACE: dataset_files_trace,
        LOVECLIM: dataset_files_loveclm,
        FAMOUS: dataset_files_famous,
    }
    
    return dataset_files_list

def find_nearest_time_index(time_array, target_time):
    """
    Finds the index of the nearest value in a time array to the target time.

    Parameters:
    - time_array (numpy.array): Array of time values.
    - target_time (int or float): Target time value.

    Returns:
    - nearest_index (int): Index of the nearest time value in the time array.
    """
    nearest_index = np.argmin(np.abs(time_array - target_time))
    return nearest_index

def plot_a_map(chosen_dataset, time_entered): 
    
    datasets = load_datasets()
    
    dataset_files = datasets[chosen_dataset]
    
    var, lat, lon, time = dataset_files
        
    time_idx = find_nearest_time_index(time, time_entered)
    
    print(f'neared idx to {time_entered} is {time_idx}')
    
    selected_frame = var[time_idx, :, :]
     
    print(np.shape(selected_frame))   
    
    # Plotting the map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    heatmap = ax.pcolormesh(lon, lat, selected_frame, cmap='YlOrBr', transform=ccrs.PlateCarree())
    plt.colorbar(heatmap, orientation='vertical', label='Mean Lightning (mm/day)')

    ax.coastlines()

    plt.title(f'Heatmap ({chosen_dataset}) of Lightning at Time {time_entered} kaBP')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Convert the plot to a Streamlit image and display it
    st.pyplot(fig)
    
    # Save plot to a BytesIO buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    return img_buffer

    
def create_download_link(file_buffer, filename, link_description, file_type='application/zip'):
    file_buffer.seek(0)  # Ensure the buffer is at the beginning
    b64 = base64.b64encode(file_buffer.read()).decode()
    # return f'<a href="data:application/zip;base64,{b64}" download="{filename}.zip">{link_description}</a>'
    return f'<a href="data:{file_type};base64,{b64}" download="{filename}">{link_description}</a>'

if generate: 
    img_buffer = plot_a_map(chosen_dataset.upper(), time_entered)
    
    img_download_url = create_download_link(img_buffer, 'plot.png', 'Download map', file_type='image/png')
    st.markdown(img_download_url, unsafe_allow_html=True)
    