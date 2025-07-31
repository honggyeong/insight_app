import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from data_utils import load_charger_data, load_support_center_data
from map_utils import create_map, add_markers, get_daegu_graph, find_route_with_pois
from cv_utils import RoadSegmentation, ARNavigation, process_frame_for_ar, haversine_distance
import cv2
import numpy as np
from PIL import Image
import time
import threading
import math

# 데이터 경로
CHARGER_PATH = 'pre_data/전국전동휠체어급속충전기표준데이터.json'
SUPPORT_CENTER_PATH = 'pre_data/전국교통약자이동지원센터정보표준데이터.json'

def df_to_locations(df, name_col, lat_col, lon_col):
    return [
        {'name': row[name_col], 'lat': float(row[lat_col]), 'lon': float(row[lon_col])}
        for _, row in df.iterrows()
    ]

st.title('대구 전동휠체어 길안내 시스템 (프로토타입)')

# 데이터 불러오기
chargers = load_charger_data(CHARGER_PATH)
support_centers = load_support_center_data(SUPPORT_CENTER_PATH)

charger_locs = df_to_locations(chargers, '시설명', '위도', '경도')
support_locs = df_to_locations(support_centers, '교통약자이동지원센터명', '위도', '경도')
poi_locs = charger_locs + support_locs

# 대구 중심 좌표
daegu_center = [35.8714, 128.6014]

# Session state 초기화
if 'start' not in st.session_state:
    st.session_state['start'] = None
if 'end' not in st.session_state:
    st.session_state['end'] = None
if 'selection_mode' not in st.session_state:
    st.session_state['selection_mode'] = 'start'
if 'current_position' not in st.session_state:
    st.session_state['current_position'] = None
if 'ar_mode' not in st.session_state:
    st.session_state['ar_mode'] = False

# 사이드바에 선택 모드 및 좌표 입력
st.sidebar.header('출발지/목적지 설정')

selection_mode = st.sidebar.radio(
    "선택 모드",
    ["출발지 선택", "목적지 선택"],
    index=0 if st.session_state['selection_mode'] == 'start' else 1
)

if selection_mode == "출발지 선택":
    st.session_state['selection_mode'] = 'start'
else:
    st.session_state['selection_mode'] = 'end'

# 좌표 직접 입력
st.sidebar.subheader('좌표 직접 입력')
lat_input = st.sidebar.number_input('위도', value=35.8714, format='%.6f')
lon_input = st.sidebar.number_input('경도', value=128.6014, format='%.6f')

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button('현재 좌표로 설정'):
        if st.session_state['selection_mode'] == 'start':
            st.session_state['start'] = (lat_input, lon_input)
            st.sidebar.success(f"출발지 설정: {lat_input:.5f}, {lon_input:.5f}")
        else:
            st.session_state['end'] = (lat_input, lon_input)
            st.sidebar.success(f"목적지 설정: {lat_input:.5f}, {lon_input:.5f}")

with col2:
    if st.button('모두 초기화'):
        st.session_state['start'] = None
        st.session_state['end'] = None
        st.session_state['route_result'] = None
        st.session_state['current_position'] = None
        st.session_state['ar_mode'] = False
        st.sidebar.success('모든 설정이 초기화되었습니다.')

# 현재 상태 표시
st.sidebar.subheader('현재 상태')
if st.session_state['start']:
    st.sidebar.write(f"**출발지:** {st.session_state['start'][0]:.5f}, {st.session_state['start'][1]:.5f}")
else:
    st.sidebar.write("**출발지:** 설정되지 않음")

if st.session_state['end']:
    st.sidebar.write(f"**목적지:** {st.session_state['end'][0]:.5f}, {st.session_state['end'][1]:.5f}")
else:
    st.sidebar.write("**목적지:** 설정되지 않음")

# 메인 지도
st.subheader('지도에서 출발지와 목적지를 선택하세요')
st.write(f"현재 선택 모드: **{selection_mode}**")

fmap = create_map(daegu_center, zoom_start=12)
add_markers(fmap, charger_locs, marker_type='blue')
add_markers(fmap, support_locs, marker_type='green')

# 기존 출발/도착 마커 표시
import folium
if st.session_state['start']:
    folium.Marker(st.session_state['start'], popup='출발지', icon=folium.Icon(color='red')).add_to(fmap)
if st.session_state['end']:
    folium.Marker(st.session_state['end'], popup='목적지', icon=folium.Icon(color='orange')).add_to(fmap)

# 지도 클릭 이벤트
map_data = st_folium(fmap, width=700, height=500, returned_objects=["last_clicked"])

if map_data and map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    
    if st.session_state['selection_mode'] == 'start':
        st.session_state['start'] = (lat, lon)
        st.success(f"출발지 선택 완료: {lat:.5f}, {lon:.5f}")
    else:
        st.session_state['end'] = (lat, lon)
        st.success(f"목적지 선택 완료: {lat:.5f}, {lon:.5f}")

# 경로탐색
if st.session_state['start'] and st.session_state['end']:
    st.subheader('경로탐색')
    
    # 경로탐색 결과를 저장할 session state
    if 'route_result' not in st.session_state:
        st.session_state['route_result'] = None
    
    if st.button('경로 탐색 시작'):
        try:
            with st.spinner('경로탐색 중... (최초 1회는 도로망 다운로드로 시간이 걸릴 수 있습니다)'):
                G = get_daegu_graph()
                route, via_poi = find_route_with_pois(G, st.session_state['start'], st.session_state['end'], poi_locs, max_dist=200)
                
                # 결과를 session state에 저장
                st.session_state['route_result'] = {
                    'route': route,
                    'via_poi': via_poi,
                    'graph': G
                }
                
                st.success('경로탐색이 완료되었습니다!')
                
        except Exception as e:
            st.error(f'경로탐색 중 오류가 발생했습니다: {str(e)}')
            st.session_state['route_result'] = None
    
    # 경로 결과 표시
    if st.session_state['route_result']:
        st.subheader('📍 경로 안내')
        
        route = st.session_state['route_result']['route']
        via_poi = st.session_state['route_result']['via_poi']
        G = st.session_state['route_result']['graph']
        
        # 경로 정보 표시
        col1, col2 = st.columns(2)
        with col1:
            st.write("**출발지:**", f"{st.session_state['start'][0]:.5f}, {st.session_state['start'][1]:.5f}")
            st.write("**목적지:**", f"{st.session_state['end'][0]:.5f}, {st.session_state['end'][1]:.5f}")
        
        with col2:
            if route:
                # 경로 거리 계산
                total_distance = 0
                for i in range(len(route) - 1):
                    edge_data = G[route[i]][route[i+1]]
                    if 'length' in edge_data[0]:
                        total_distance += edge_data[0]['length']
                
                st.write("**총 거리:**", f"{total_distance:.1f}m")
                st.write("**경유지:**", via_poi['name'] if via_poi else "없음")
        
        # 경로 지도 표시
        st.write("**경로 지도:**")
        fmap2 = create_map(daegu_center, zoom_start=12)
        add_markers(fmap2, charger_locs, marker_type='blue')
        add_markers(fmap2, support_locs, marker_type='green')
        
        # 출발/도착 마커
        folium.Marker(st.session_state['start'], popup='출발지', icon=folium.Icon(color='red')).add_to(fmap2)
        folium.Marker(st.session_state['end'], popup='목적지', icon=folium.Icon(color='orange')).add_to(fmap2)
        
        # 경로 polyline
        if route:
            route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            folium.PolyLine(route_coords, color='purple', weight=6, opacity=0.7).add_to(fmap2)
        
        # 경유지 마커
        if via_poi:
            folium.Marker([via_poi['lat'], via_poi['lon']], popup=f"경유: {via_poi['name']}", icon=folium.Icon(color='cadetblue')).add_to(fmap2)
        
        st_folium(fmap2, width=700, height=500)
        
        # 경로 안내 메시지
        if via_poi:
            st.info(f"🎯 **경로 안내**: 출발지에서 {via_poi['name']} (경유지)를 거쳐 목적지까지 이동하세요.")
        else:
            st.info("🎯 **경로 안내**: 경로가 이미 충전기/지원센터 근처를 지나므로 최단경로로 이동하세요.")
        
        # 증강현실 모드 시작
        st.subheader('🎥 증강현실 길안내')
        
        # 현재 위치 설정
        st.write("**현재 위치 설정:**")
        current_lat = st.number_input('현재 위도', value=st.session_state['start'][0], format='%.6f', key='current_lat')
        current_lon = st.number_input('현재 경도', value=st.session_state['start'][1], format='%.6f', key='current_lon')
        
        if st.button('현재 위치 업데이트'):
            st.session_state['current_position'] = (current_lat, current_lon)
            st.success(f"현재 위치 업데이트: {current_lat:.5f}, {current_lon:.5f}")
        
        # AR 모드 선택
        ar_mode = st.radio(
            "증강현실 모드 선택",
            ["사진 업로드", "실시간 카메라"],
            index=0
        )
        
        if ar_mode == "사진 업로드":
            st.write("**사진을 업로드하여 증강현실 안내를 받아보세요:**")
            
            uploaded_file = st.file_uploader("카메라 사진 업로드", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None and st.session_state['current_position']:
                # 이미지 로드
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # BGR로 변환 (OpenCV 형식)
                if len(image_np.shape) == 3:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                # AR 처리
                processed_frame = process_frame_for_ar(image_bgr, st.session_state['current_position'], st.session_state['end'])
                
                # 결과 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**원본 영상**")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.write("**증강현실 안내**")
                    # BGR을 RGB로 변환하여 표시
                    processed_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(processed_image, use_container_width=True)
                
                # 컴퓨터비전 분석 결과
                st.subheader('🔍 컴퓨터비전 분석')
                road_seg = RoadSegmentation()
                road_mask, sidewalk_mask = road_seg.segment_road_sidewalk(image_bgr)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**원본 영상**")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.write("**차도 마스크**")
                    st.image(road_mask, use_container_width=True, channels="GRAY")
                
                with col3:
                    st.write("**보행로 마스크**")
                    st.image(sidewalk_mask, use_container_width=True, channels="GRAY")
        
        elif ar_mode == "실시간 카메라":
            st.write("**실시간 카메라를 통해 증강현실 길안내를 받아보세요:**")
            
            if st.session_state['current_position']:
                # Streamlit 내장 카메라 사용
                camera_input = st.camera_input("실시간 카메라", key="realtime_camera")
                
                if camera_input is not None:
                    # 이미지를 OpenCV 형식으로 변환
                    image = Image.open(camera_input)
                    image_np = np.array(image)
                    
                    # BGR로 변환 (OpenCV 형식)
                    if len(image_np.shape) == 3:
                        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                    
                    # AR 처리
                    processed_frame = process_frame_for_ar(image_bgr, st.session_state['current_position'], st.session_state['end'])
                    
                    # 결과 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**원본 영상**")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.write("**증강현실 안내**")
                        # BGR을 RGB로 변환하여 표시
                        processed_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st.image(processed_image, use_container_width=True)
                    
                    # 컴퓨터비전 분석 결과
                    st.subheader('🔍 실시간 컴퓨터비전 분석')
                    road_seg = RoadSegmentation()
                    road_mask, sidewalk_mask = road_seg.segment_road_sidewalk(image_bgr)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**원본 영상**")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.write("**차도 마스크**")
                        st.image(road_mask, use_container_width=True, channels="GRAY")
                    
                    with col3:
                        st.write("**보행로 마스크**")
                        st.image(sidewalk_mask, use_container_width=True, channels="GRAY")
                    
                    # 실시간 안내 정보
                    distance = haversine_distance(
                        st.session_state['current_position'][0], 
                        st.session_state['current_position'][1], 
                        st.session_state['end'][0], 
                        st.session_state['end'][1]
                    )
                    
                    direction = math.degrees(math.atan2(
                        st.session_state['end'][0] - st.session_state['current_position'][0], 
                        st.session_state['end'][1] - st.session_state['current_position'][1]
                    ))
                    direction = (direction + 360) % 360
                    
                    st.subheader('🎯 실시간 안내 정보')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("거리", f"{distance:.0f}m")
                    with col2:
                        st.metric("방향", f"{direction:.0f}°")
                    with col3:
                        if distance < 10:
                            st.success("목적지 도착!")
                        elif distance < 50:
                            st.warning("곧 도착합니다")
                        else:
                            st.info("경로를 따라 이동하세요")
                
                else:
                    st.info("카메라를 켜서 실시간 증강현실 안내를 시작하세요.")
            else:
                st.warning("먼저 현재 위치를 설정해주세요.")
        
        # 경로 초기화 버튼
        if st.button('새로운 경로 탐색'):
            st.session_state['route_result'] = None
            st.rerun()
else:
    st.info('출발지와 목적지를 모두 설정한 후 경로탐색을 시작하세요.')

st.markdown('---')
st.write('※ 실시간 증강현실 안내 시스템이 구현되었습니다! 🎥✨')