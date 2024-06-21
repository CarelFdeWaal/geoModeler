import xml.etree.ElementTree as ET
import requests
import os
import argparse
import gpxpy
import numpy as np
import rasterio
from stl import mesh
from dotenv import load_dotenv

def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
    
    return points

def calculate_bounding_box(points):
    min_lat = min(point[0] for point in points)
    max_lat = max(point[0] for point in points)
    min_lon = min(point[1] for point in points)
    max_lon = max(point[1] for point in points)
    
    return min_lat, max_lat, min_lon, max_lon



def get_elevation_data(min_lat, max_lat, min_lon, max_lon):
    api_key = os.getenv('OPENTOPO_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set the OPENTOPO_API_KEY environment variable.")

    opentopo_url = 'https://portal.opentopography.org/API/globaldem'
    params = {
        'demtype': 'SRTMGL1',  # Specify the DEM type, e.g., SRTMGL1, SRTMGL3
        'south': min_lat,
        'north': max_lat,
        'west': min_lon,
        'east': max_lon,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }

    response = requests.get(opentopo_url, params=params)
    if response.status_code == 200:
        with open('elevation_data.tif', 'wb') as file:
            file.write(response.content)
        return 'elevation_data.tif'
    else:
        print(f"Error message: {response.text}")
        raise Exception(f"Error fetching elevation data: {response.status_code}")


def create_3d_model(elevation_data_file):
    with rasterio.open(elevation_data_file) as dataset:
        elevation_data = dataset.read(1)
    
    # Normalize and scale the elevation data
    z_scale = 1  # Adjust this scale as necessary
    x_scale = y_scale = 150 / max(elevation_data.shape)
    
    # Create a grid of points
    x = np.linspace(0, 150, elevation_data.shape[1])
    y = np.linspace(0, 150, elevation_data.shape[0])
    x, y = np.meshgrid(x, y)
    
    z = elevation_data * z_scale
    
    # Create vertices and faces for the 3D mesh
    vertices = np.zeros((elevation_data.size, 3))
    vertices[:, 0] = x.ravel()
    vertices[:, 1] = y.ravel()
    vertices[:, 2] = z.ravel()
    
    faces = []
    for i in range(elevation_data.shape[0] - 1):
        for j in range(elevation_data.shape[1] - 1):
            idx = i * elevation_data.shape[1] + j
            faces.append([idx, idx + 1, idx + elevation_data.shape[1]])
            faces.append([idx + 1, idx + elevation_data.shape[1] + 1, idx + elevation_data.shape[1]])
    
    faces = np.array(faces)
    
    # Create the mesh
    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]
    
    # Save the mesh to an STL file
    terrain_mesh.save('terrain.stl')
    
    print("3D model created and saved as 'terrain.stl'")


def main():
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(description='Parse a GPX file and get elevation data for the bounding box.')
    parser.add_argument('gpx_file', type=str, help='Path to the GPX file')

    args = parser.parse_args()
    points = parse_gpx(args.gpx_file)
    min_lat, max_lat, min_lon, max_lon = calculate_bounding_box(points)
    
    print(f"Bounding Box: South={min_lat}, North={max_lat}, West={min_lon}, East={max_lon}")
    print(f"API Key: {os.getenv('OPENTOPO_API_KEY')}")
    
    elevation_data_file = get_elevation_data(min_lat, max_lat, min_lon, max_lon)
    create_3d_model(elevation_data_file)

if __name__ == "__main__":
    main()

