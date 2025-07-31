import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import math
import threading
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class RoadSegmentation:
    """차도와 보행로를 구분하는 클래스"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def segment_road_sidewalk(self, frame):
        """입력 프레임에서 차도/보행로를 분리"""
        # BGR to HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 차도 (회색/검은색) 마스크
        lower_road = np.array([0, 0, 30])
        upper_road = np.array([180, 30, 150])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        
        # 보행로 (밝은색/흰색) 마스크
        lower_sidewalk = np.array([0, 0, 150])
        upper_sidewalk = np.array([180, 30, 255])
        sidewalk_mask = cv2.inRange(hsv, lower_sidewalk, upper_sidewalk)
        
        return road_mask, sidewalk_mask
    
    def detect_lanes(self, frame):
        """차선 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 관심 영역 설정 (화면 하단 절반)
        height, width = edges.shape
        roi_vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough 변환으로 직선 감지
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        
        return lines

class LocationTracker:
    """GPS와 컴퓨터비전을 결합한 위치 추적 클래스"""
    
    def __init__(self):
        self.landmark_database = {}  # 지형지물 데이터베이스
        self.gps_history = []  # GPS 위치 히스토리
        self.vision_history = []  # 비전 기반 위치 히스토리
        self.fused_location = None  # 융합된 위치
        
    def add_landmark(self, name, lat, lon, features):
        """지형지물 데이터베이스에 추가"""
        self.landmark_database[name] = {
            'lat': lat,
            'lon': lon,
            'features': features
        }
    
    def detect_landmarks_in_image(self, frame):
        """이미지에서 지형지물 감지"""
        landmarks = []
        
        # 건물 감지 (간단한 엣지 검출 기반)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 충분히 큰 객체만
                x, y, w, h = cv2.boundingRect(contour)
                landmarks.append({
                    'type': 'building',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
        
        return landmarks
    
    def estimate_position_from_landmarks(self, detected_landmarks, gps_estimate):
        """감지된 지형지물을 기반으로 위치 추정"""
        if not detected_landmarks:
            return gps_estimate
        
        # GPS 추정치를 중심으로 주변 지형지물 검색
        search_radius = 0.001  # 약 100m
        nearby_landmarks = []
        
        for name, landmark in self.landmark_database.items():
            dist = haversine_distance(
                gps_estimate[0], gps_estimate[1],
                landmark['lat'], landmark['lon']
            )
            if dist < 100:  # 100m 이내
                nearby_landmarks.append((name, landmark, dist))
        
        if not nearby_landmarks:
            return gps_estimate
        
        # 가장 가까운 지형지물 기반으로 위치 보정
        nearest_landmark = min(nearby_landmarks, key=lambda x: x[2])
        return (nearest_landmark[1]['lat'], nearest_landmark[1]['lon'])
    
    def update_gps_location(self, lat, lon, accuracy=None):
        """GPS 위치 업데이트"""
        self.gps_history.append({
            'lat': lat,
            'lon': lon,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        # 최근 10개 GPS 위치만 유지
        if len(self.gps_history) > 10:
            self.gps_history.pop(0)
    
    def update_vision_location(self, frame, gps_estimate):
        """비전 기반 위치 업데이트"""
        landmarks = self.detect_landmarks_in_image(frame)
        vision_estimate = self.estimate_position_from_landmarks(landmarks, gps_estimate)
        
        self.vision_history.append({
            'lat': vision_estimate[0],
            'lon': vision_estimate[1],
            'landmarks': landmarks,
            'timestamp': time.time()
        })
        
        # 최근 10개 비전 위치만 유지
        if len(self.vision_history) > 10:
            self.vision_history.pop(0)
        
        return vision_estimate
    
    def fuse_locations(self, gps_weight=0.7, vision_weight=0.3):
        """GPS와 비전 위치 융합"""
        if not self.gps_history or not self.vision_history:
            return None
        
        # 최신 GPS 위치
        latest_gps = self.gps_history[-1]
        
        # 최신 비전 위치
        latest_vision = self.vision_history[-1]
        
        # 가중 평균으로 위치 융합
        fused_lat = gps_weight * latest_gps['lat'] + vision_weight * latest_vision['lat']
        fused_lon = gps_weight * latest_gps['lon'] + vision_weight * latest_vision['lon']
        
        self.fused_location = (fused_lat, fused_lon)
        return self.fused_location
    
    def get_trajectory(self):
        """이동 궤적 반환"""
        trajectory = []
        
        # GPS 궤적
        gps_traj = [(p['lat'], p['lon']) for p in self.gps_history]
        
        # 비전 궤적
        vision_traj = [(p['lat'], p['lon']) for p in self.vision_history]
        
        # 융합 궤적
        if self.fused_location:
            trajectory.append(self.fused_location)
        
        return {
            'gps': gps_traj,
            'vision': vision_traj,
            'fused': trajectory
        }

class ARNavigation:
    """증강현실 길안내 클래스"""
    
    def __init__(self):
        self.font_path = self._get_font_path()
        self.location_tracker = LocationTracker()
        
    def _get_font_path(self):
        """시스템 폰트 경로 찾기"""
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        elif system == "Windows":
            return "C:/Windows/Fonts/malgun.ttf"
        else:  # Linux
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    def put_korean_text(self, img, text, position, font_size=30, color=(255, 255, 255)):
        """한글 텍스트를 이미지에 추가"""
        try:
            # PIL 이미지로 변환
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 폰트 로드
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except:
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # 텍스트 그리기
            draw.text(position, text, font=font, fill=color)
            
            # OpenCV 이미지로 변환
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
        except Exception as e:
            print(f"한글 텍스트 렌더링 오류: {e}")
            # 폴백: 기본 OpenCV 텍스트
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return img
    
    def draw_direction_arrow(self, frame, direction_angle, center=(320, 240)):
        """방향 화살표 그리기"""
        # 화살표 길이와 각도
        arrow_length = 80
        angle_rad = math.radians(direction_angle)
        
        # 화살표 끝점 계산
        end_x = int(center[0] + arrow_length * math.cos(angle_rad))
        end_y = int(center[1] - arrow_length * math.sin(angle_rad))
        
        # 화살표 그리기
        cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), 5, tipLength=0.3)
        
        # 방향 텍스트
        direction_text = f"방향: {direction_angle:.0f}°"
        frame = self.put_korean_text(frame, direction_text, (10, 30), 25, (255, 255, 255))
        
        return frame
    
    def add_navigation_info(self, frame, distance, direction_angle, current_pos, target_pos):
        """길안내 정보 추가"""
        # 거리 정보
        if distance < 10:
            message = "목적지 도착!"
            color = (0, 255, 0)  # 녹색
        elif distance < 50:
            message = f"곧 도착합니다 ({distance:.0f}m)"
            color = (0, 255, 255)  # 노란색
        else:
            message = f"경로를 따라 이동하세요 ({distance:.0f}m)"
            color = (255, 255, 255)  # 흰색
        
        # 메시지 표시
        frame = self.put_korean_text(frame, message, (10, 70), 25, color)
        
        # 방향 화살표
        frame = self.draw_direction_arrow(frame, direction_angle)
        
        # 현재 위치 정보
        pos_text = f"현재: ({current_pos[0]:.5f}, {current_pos[1]:.5f})"
        frame = self.put_korean_text(frame, pos_text, (10, 110), 20, (200, 200, 200))
        
        return frame
    
    def add_location_tracking_info(self, frame, gps_pos, vision_pos, fused_pos):
        """위치 추적 정보 추가"""
        # GPS 위치
        gps_text = f"GPS: ({gps_pos[0]:.5f}, {gps_pos[1]:.5f})"
        frame = self.put_korean_text(frame, gps_text, (10, 140), 18, (0, 255, 255))
        
        # 비전 위치
        vision_text = f"비전: ({vision_pos[0]:.5f}, {vision_pos[1]:.5f})"
        frame = self.put_korean_text(frame, vision_text, (10, 160), 18, (255, 0, 255))
        
        # 융합 위치
        fused_text = f"융합: ({fused_pos[0]:.5f}, {fused_pos[1]:.5f})"
        frame = self.put_korean_text(frame, fused_text, (10, 180), 18, (0, 255, 0))
        
        return frame

class RealTimeAR:
    """실시간 증강현실 처리 클래스"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.road_seg = RoadSegmentation()
        self.ar_nav = ARNavigation()
        
    def start_camera(self, camera_index=0):
        """카메라 시작"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("카메라를 열 수 없습니다.")
        
        self.is_running = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def stop_camera(self):
        """카메라 중지"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        """현재 프레임 반환"""
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def process_frame_ar(self, frame, current_pos, target_pos, route_info=None):
        """프레임에 AR 정보 추가"""
        if frame is None:
            return None
            
        # 거리와 방향 계산
        distance = haversine_distance(current_pos[0], current_pos[1], target_pos[0], target_pos[1])
        direction = math.degrees(math.atan2(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]))
        direction = (direction + 360) % 360
        
        # AR 정보 추가
        processed_frame = self.ar_nav.add_navigation_info(frame, distance, direction, current_pos, target_pos)
        
        # 차도/보행로 세그멘테이션 (선택적)
        road_mask, sidewalk_mask = self.road_seg.segment_road_sidewalk(frame)
        
        # 차선 감지 (선택적)
        lanes = self.road_seg.detect_lanes(frame)
        if lanes is not None:
            for line in lanes:
                x1, y1, x2, y2 = line[0]
                cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return processed_frame
    
    def run_realtime_ar(self, current_pos, target_pos, callback=None):
        """실시간 AR 실행"""
        try:
            self.start_camera()
            
            while self.is_running:
                frame = self.get_frame()
                if frame is not None:
                    processed_frame = self.process_frame_ar(frame, current_pos, target_pos)
                    
                    if callback:
                        callback(processed_frame)
                    
                    # ESC 키로 종료
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                        
        except Exception as e:
            print(f"실시간 AR 오류: {e}")
        finally:
            self.stop_camera()

def haversine_distance(lat1, lon1, lat2, lon2):
    """두 지점 간의 거리를 계산 (haversine 공식)"""
    R = 6371000  # 지구 반지름 (미터)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def process_frame_for_ar(frame, current_pos, target_pos):
    """단일 프레임 AR 처리 (기존 함수 유지)"""
    ar_nav = ARNavigation()
    road_seg = RoadSegmentation()
    
    # 거리와 방향 계산
    distance = haversine_distance(current_pos[0], current_pos[1], target_pos[0], target_pos[1])
    direction = math.degrees(math.atan2(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]))
    direction = (direction + 360) % 360
    
    # AR 정보 추가
    processed_frame = ar_nav.add_navigation_info(frame, distance, direction, current_pos, target_pos)
    
    return processed_frame

def process_frame_with_location_tracking(frame, gps_pos, target_pos):
    """위치 추적이 포함된 프레임 처리"""
    ar_nav = ARNavigation()
    road_seg = RoadSegmentation()
    
    # GPS 위치 업데이트
    ar_nav.location_tracker.update_gps_location(gps_pos[0], gps_pos[1])
    
    # 비전 기반 위치 추정
    vision_pos = ar_nav.location_tracker.update_vision_location(frame, gps_pos)
    
    # 위치 융합
    fused_pos = ar_nav.location_tracker.fuse_locations()
    
    if fused_pos is None:
        fused_pos = gps_pos
    
    # 거리와 방향 계산 (융합된 위치 사용)
    distance = haversine_distance(fused_pos[0], fused_pos[1], target_pos[0], target_pos[1])
    direction = math.degrees(math.atan2(target_pos[0] - fused_pos[0], target_pos[1] - fused_pos[1]))
    direction = (direction + 360) % 360
    
    # AR 정보 추가
    processed_frame = ar_nav.add_navigation_info(frame, distance, direction, fused_pos, target_pos)
    
    # 위치 추적 정보 추가
    processed_frame = ar_nav.add_location_tracking_info(frame, gps_pos, vision_pos, fused_pos)
    
    return processed_frame, fused_pos