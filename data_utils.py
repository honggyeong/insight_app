import pandas as pd
import json

def load_charger_data(filepath):
    """전국전동휠체어급속충전기 데이터에서 대구 지역만 필터링 (공공데이터 표준 구조)"""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data['records'])
    print('충전기 데이터 컬럼:', df.columns.tolist())
    print('충전기 데이터 샘플:', df.head())
    df_daegu = df[df['시도명'].str.contains('대구')]
    return df_daegu

def load_support_center_data(filepath):
    """전국교통약자이동지원센터 데이터에서 대구 지역만 필터링 (공공데이터 표준 구조)"""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data['records'])
    print('지원센터 데이터 컬럼:', df.columns.tolist())
    print('지원센터 데이터 샘플:', df.head())
    df_daegu = df[df['소재지도로명주소'].str.contains('대구')]
    return df_daegu