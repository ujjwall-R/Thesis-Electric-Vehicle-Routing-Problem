#!/usr/bin/env python
# coding: utf-8

# In[128]:


get_ipython().system('pip install requests')
get_ipython().system('pip install numpy pandas')
get_ipython().system('pip install python-dotenv')


# ### Function to calculate road distance b/w any 2 loactions

# In[129]:


import numpy as np
import pandas as pd
import time


# In[130]:


from dotenv import load_dotenv
import os
load_dotenv()
openrouteservice_api_key=os.getenv("openrouteservice_api_key")


# In[131]:


import requests

def get_road_distance_and_time(start_coords, end_coords, profile='driving-car'):
    api_key=openrouteservice_api_key
    base_url = 'https://api.openrouteservice.org/v2/directions/'
    url = f'{base_url}{profile}'
    params = {
        'api_key': api_key,
        'start': f'{start_coords[1]},{start_coords[0]}',  #order: longitude,latitude
        'end': f'{end_coords[1]},{end_coords[0]}',
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        print(data)
        if 'features' in data and len(data['features']) > 0:
            distance = data['features'][0]['properties']['segments'][0]['distance']
            timebwNodes = data['features'][0]['properties']['segments'][0]['duration']
            return [distance,timebwNodes]
        else:
            print('No route found.')
            return None
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
        return None


# In[132]:


# dist = get_road_distance((12.9929356,77.5951078),(12.9694440,77.7499922))
# print(dist)


# ### Fetch Data

# In[133]:


import requests

def get_coordinates(place_name):
    base_url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': place_name,
        'format': 'json',
        'limit': 1,
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    # print(data)
    if data:
        location = (float(data[0]['lat']), float(data[0]['lon']))
        return location
    else:
        return None

charging_stations = [get_coordinates('charging_station, Bangalore Indira Nagar'),get_coordinates('charging_station, Bangalore Electronic city')]

bus_stop_coordinates = []
ind = 0
while len(bus_stop_coordinates) < 13:
    bus_stop = get_coordinates(f'Bus Stop {ind}, Bangalore')
    ind=ind+1
    if bus_stop:
        bus_stop_coordinates.append(bus_stop)
    bus_stop_coordinates=list(set(bus_stop_coordinates))

car_workshop = get_coordinates('Car Workshop, Bangalore')


# In[134]:


bus_stop_coordinates


# In[135]:


car_workshop


# In[136]:


charging_stations


# ### Sample Data
# - car_workshop : list of depot (0) 
# - chargind_stations : BSS 
# - bust_stop_coordinates : customers 
# 
# ### Sample Space
# - [depot, ...BSS, ...customers, depot]

# In[137]:


nodes = [car_workshop]+ charging_stations + bus_stop_coordinates + [car_workshop]


# ### distance matrix

# In[138]:


number_of_node = len(nodes)
distance_matrix = np.zeros((number_of_node, number_of_node))
distance_matrix = pd.DataFrame(distance_matrix)
time_matrix = np.zeros((number_of_node, number_of_node))
time_matrix = pd.DataFrame(time_matrix)

for i in range(number_of_node):
    for j in range(number_of_node):
            distTime=get_road_distance_and_time(nodes[i],nodes[j])
            distance_matrix.iloc[i, j] = distTime[0]/1000
            time_matrix.iloc[i,j]=distTime[1]/3600
            time.sleep(2)


# In[139]:


distance_matrix


# In[140]:


time_matrix


# In[141]:


distance_matrix.to_csv('distance_matrix.csv', index=False)

print("DataFrame saved to 'distance_matrix.csv'")


# In[142]:


time_matrix.to_csv('time_matrix.csv', index=False)

print("DataFrame saved to 'time_matrix.csv'")


# #### EV : Volvo C40 Recharge
# - Mileage : 530 km/full charge
# - Battery capacity : 78 kWh
# - Charging time : 27 min
# - Power : 300 kW

# In[143]:


mileage = 530 #km/full charge
battery_capacity = 78 #kWh
battery_expenditure_per_km = battery_capacity/mileage


# In[144]:


battery_expenditure_matrix = np.zeros((number_of_node, number_of_node))
battery_expenditure_matrix = pd.DataFrame(battery_expenditure_matrix)

for i in range(number_of_node):
    for j in range(number_of_node):
            battery_expenditure_matrix.iloc[i,j]=distance_matrix.iloc[i,j]*battery_expenditure_per_km


# In[145]:


battery_expenditure_matrix


# In[146]:


battery_expenditure_matrix.to_csv('battery_expenditure_matrix.csv', index=False)

print("DataFrame saved to 'battery_expenditure_matrix.csv'")


# In[147]:


save_nodes=pd.DataFrame(nodes)
save_nodes.to_csv('nodes.csv', index=False)

print("DataFrame saved to 'nodes.csv'")


# 
