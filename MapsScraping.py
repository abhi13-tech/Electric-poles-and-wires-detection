import requests
import os
import osmnx as ox
from shapely.geometry import LineString
import numpy as np
import math
import pyproj
from shapely.ops import transform, linemerge, unary_union
from shapely.geometry import LineString
import matplotlib.pyplot as plt



# Constants
API_KEY = 'AIzaSyABlU3q4ePCLUTWdakOVBoxHaJB3o0QMc0'
BASE_IMAGE_URL = 'https://maps.googleapis.com/maps/api/streetview'
BASE_METADATA_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
BASE_DIRECTIONS_URL = 'https://maps.googleapis.com/maps/api/directions/json'

#Montclair
#center_point = (40.811474, -74.214474)

#Oakland
center_point = (37.957511, -122.067032)

step_m = 20   
SAVE_DIR = 'street_view_route_images'
FOV = 90 
PITCH = 20
RADIUS = 30
IMAGE_SIZE = '640x640'

def compute_heading(start, end):
    lon1, lat1 = start
    lon2, lat2 = end
    angle = math.atan2(lon2 - lon1, lat2 - lat1)
    return (math.degrees(angle) + 360) % 360

def get_street_view_metadata(lat, lng, heading): 
    """Check if Street View is available for the given parameters.""" 
    params = { 
        'location': f'{lat},{lng}', 
        'heading': heading, 
        'fov': FOV, 
        'pitch': PITCH, 
        'radius': RADIUS, 
        'key': API_KEY 
        
        } 
    response = requests.get(BASE_METADATA_URL, params=params) 
    return response.json()

def download_street_view_image(lat, lng, heading, save_path): 
    """Download Street View image for the given parameters.""" 
    params = { 
        'location': f'{lat},{lng}', 
        'heading': heading, 
        'fov': FOV, 'pitch': PITCH, 
        'size': IMAGE_SIZE, 
        'radius': RADIUS, 
        'key': API_KEY 
        } 
    response = requests.get(BASE_IMAGE_URL, params=params) 
    if response.status_code == 200: 
        with open(save_path, 'wb') as f: 
            f.write(response.content) 
            print(f"Saved image: {save_path}") 
    else: 
        print(f"Failed to get image for heading {heading}.")

def get_surrounding_street_views(location): 
    lat, lng = location 
    
    nearest_edge = ox.distance.nearest_edges(G, X=lng, Y=lat)
    u, v, key = nearest_edge
    data = G.get_edge_data(u, v)[0]
    line = data.get("geometry")
    start = line.coords[0]
    next_pt = line.coords[1]
    road_heading = compute_heading(start, next_pt)
    sidewalk_frontright = (road_heading + 30) % 360
    sidewalk_right = (road_heading + 90) % 360
    sidewalk_backright = (road_heading + 150) % 360
    sidewalk_frontleft = (road_heading + 330) % 360
    sidewalk_left = (road_heading + 270) % 360
    sidewalk_backleft = (road_heading + 210) % 360
    HEADINGS = [sidewalk_frontright, sidewalk_right, sidewalk_backright, sidewalk_frontleft, sidewalk_left, sidewalk_backleft]
    for heading in HEADINGS: 
        metadata = get_street_view_metadata(lat, lng, heading + 90) 
        if metadata.get('status') == 'OK': 
            filename = f"pano_{lat}_{lng}_heading_{heading}.jpg" 
            save_path = os.path.join(SAVE_DIR, filename) 
            download_street_view_image(lat, lng, heading, save_path) 
        else: 
            print(f"No Street View available at heading {heading}.")

def dedup_points(points, tolerance_m=5):
    from shapely.strtree import STRtree
    from shapely.geometry import Point
    pts = [Point(lon, lat) for lat, lon in points]
    pts_m = [transform(project, p) for p in pts]
    tree = STRtree(pts_m)
    unique = []
    seen = set()
    for i, p in enumerate(pts_m):
        if i in seen:
            continue
        nearby = [j for j in tree.query(p.buffer(tolerance_m))]
        seen.update(nearby)
        unique.append(points[i])
    return unique

os.makedirs(SAVE_DIR, exist_ok=True)
G = ox.graph_from_point(center_point, dist=1000, network_type='drive', simplify=False)
nodes, edges = ox.graph_to_gdfs(G)

coords = list(zip(nodes['y'], nodes['x']))  # lat, lng
for u, v, key, data in G.edges(keys=True, data=True):
    if 'geometry' not in data:
        point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
        point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
        data['geometry'] = LineString([point_u, point_v])

project = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform
reproject = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True).transform
all_lines = [geom for geom in edges.geometry if geom is not None]
merged = linemerge(unary_union(all_lines))

#change it to a list
if merged.geom_type == "MultiLineString":
    merged = [merged]

sampled_points = []

for line in merged:
    #coordinates to meters
    line_m = transform(project, line)
    length_m = line_m.length
    num_samples = int(np.floor(length_m / step_m))
    
    for i in range(num_samples + 1):
        point_m = line_m.interpolate(i * step_m)
        point_latlon = transform(reproject, point_m)
        sampled_points.append((point_latlon.y, point_latlon.x))

sampled_points = dedup_points(sampled_points)
print(f"Sampled {len(sampled_points)} points spaced ~{step_m}m apart along merged roads.")

fig, ax = ox.plot_graph(G, show=False, close=False)
ys, xs = zip(*sampled_points)
ax.scatter(xs, ys, s=20, marker='o')
plt.show()



for LOCATION in list(set(sampled_points)):
    get_surrounding_street_views(LOCATION)
