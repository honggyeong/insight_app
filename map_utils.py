import folium
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import numpy as np
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """두 지점 간의 거리를 계산 (haversine 공식)"""
    R = 6371000  # 지구 반지름 (미터)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def create_map(center, zoom_start=13):
    """중심좌표로 folium 지도 생성"""
    return folium.Map(location=center, zoom_start=zoom_start)

def add_markers(fmap, locations, marker_type):
    """지도에 마커 추가 (충전기/지원센터/기타)"""
    for loc in locations:
        folium.Marker([loc['lat'], loc['lon']], popup=loc['name'], icon=folium.Icon(color=marker_type)).add_to(fmap)

def get_daegu_graph():
    """대구시 도로망 그래프 다운로드"""
    G = ox.graph_from_place('Daegu, South Korea', network_type='walk')
    return G

def find_nearest_node(G, point):
    """(lat, lon) 좌표를 그래프의 가장 가까운 노드로 변환"""
    return ox.nearest_nodes(G, point[1], point[0])

def find_route_with_pois(G, start, end, pois, max_dist=200):
    """
    출발~도착 최단경로를 찾되, 경로가 POI(충전기/지원센터) 근처(max_dist m 이내)를 최소 1회 이상 지나가도록 경로탐색
    - G: osmnx 그래프
    - start, end: (lat, lon)
    - pois: [{'lat':, 'lon':, 'name':}, ...]
    - max_dist: 경로와 POI의 최대 거리(m)
    """
    start_node = find_nearest_node(G, start)
    end_node = find_nearest_node(G, end)
    # 1. 최단경로
    route = nx.shortest_path(G, start_node, end_node, weight='length')
    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
    # 2. 경로가 POI 근처를 지나는지 확인
    for poi in pois:
        min_dist = min([haversine_distance(poi['lat'], poi['lon'], lat, lon) for lat, lon in route_coords])
        if min_dist <= max_dist:
            # 이미 경로가 POI 근처를 지난다
            return route, None
    # 3. 경로가 POI 근처를 지나지 않으면, 가장 가까운 POI를 경유지로 추가
    min_poi_dist = float('inf')
    nearest_poi = None
    for poi in pois:
        # 출발지에서 POI까지 + POI에서 목적지까지의 거리
        poi_node = find_nearest_node(G, (poi['lat'], poi['lon']))
        try:
            route1 = nx.shortest_path(G, start_node, poi_node, weight='length')
            route2 = nx.shortest_path(G, poi_node, end_node, weight='length')
            total_dist = sum(G[route1[i]][route1[i+1]][0]['length'] for i in range(len(route1)-1)) + \
                        sum(G[route2[i]][route2[i+1]][0]['length'] for i in range(len(route2)-1))
            if total_dist < min_poi_dist:
                min_poi_dist = total_dist
                nearest_poi = poi
        except nx.NetworkXNoPath:
            continue
    if nearest_poi:
        # 경유지 포함 경로 계산
        poi_node = find_nearest_node(G, (nearest_poi['lat'], nearest_poi['lon']))
        route1 = nx.shortest_path(G, start_node, poi_node, weight='length')
        route2 = nx.shortest_path(G, poi_node, end_node, weight='length')
        # 경유지 노드 중복 제거
        route = route1[:-1] + route2
        return route, nearest_poi
    return route, None