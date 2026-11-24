# -*- coding: utf-8 -*-
"""
HP 모형 ver 3.0 — 2단계 : 침수위험등급 산정 (hp_risk)
====================================================================
건물 객체·도로 노드별 위험인자를 구성하고 위험지수(Distance)와
4단계 침수위험등급을 산정한다. (ver 2.0 HP_건물_V3.ipynb · HP_도로.ipynb 의 위험등급 산정부)

기능
  [G-1] 건물 객체별 위험인자 구성
        Hazard        = 용도가중치(weight_word) + 지하침수위험(under_risk)
        Vulnerability = 내용물 취약도(cont_vul)
        Exposure      = 노출인구지표(alllbl)
  [G-2] 도로 노드별 위험인자 구성
        Hazard        = 침수심 기반 손상가중치(WEIGHT)
        Vulnerability = 교통지표(TI)
        Exposure      = 차로수(LANES)
  [G-3] 표준화 후 위험지수(Distance) 산정
        Distance = || StandardScaler(Hazard, Vulnerability, Exposure) ||_2
  [G-4] 위험지수 군집화(KMeans)를 통한 4단계 침수위험등급 산정
        1 매우낮음 / 2 낮음 / 3 위험 / 4 매우위험
  [G-5] 행정동별 위험등급 분포 집계

출력 (시간단계별 폴더, 파일명은 ver 2.0 과 동일)
  - H_(t)MIN_results_building_risk.csv : 건물 객체별 위험지수·위험등급  (ver 2.0 열 + Grade)
  - H_(t)MIN_results_road_risk.csv     : 도로 노드별 위험지수·위험등급  (ver 3.0 신규)
  - H_(t)MIN_region_risk_grade.csv     : 행정동별 건물·도로 위험등급 분포 (ver 3.0 신규)

계산식과 상수는 ver 2.0 과 동일하며 변경하지 않았다.
"""
import os
import pickle
import numpy as np
import pandas as pd
import warnings

import hp_config as cfg
from hp_common import load_csv_safe

warnings.filterwarnings('ignore')

RISK_LEVEL_MAP = {3: '매우위험', 2: '위험', 1: '낮음', 0: '매우낮음'}
FEATURES = ['Hazard', 'Vulnerability', 'Exposure']


# =============================================================================
# 학습모델 적재
# =============================================================================
def load_risk_models():
    """위험등급 산정용 표준화 스케일러와 군집모형(건물·도로)을 적재한다."""
    with open(cfg.P_BD_SCALER, 'rb') as f:
        bd_scaler = pickle.load(f)
    with open(cfg.P_BD_KMEANS, 'rb') as f:
        bd_kmeans = pickle.load(f)
    with open(cfg.P_RD_SCALER, 'rb') as f:
        rd_scaler = pickle.load(f)
    with open(cfg.P_RD_KMEANS, 'rb') as f:
        rd_kmeans = pickle.load(f)
    return {'bd_scaler': bd_scaler, 'bd_kmeans': bd_kmeans,
            'rd_scaler': rd_scaler, 'rd_kmeans': rd_kmeans}


# =============================================================================
# [G-3] ~ [G-4] 위험지수 산정 및 등급 군집화 (건물·도로 공통)
# =============================================================================
def compute_risk_grade(df, scaler, kmeans_model):
    """Hazard/Vulnerability/Exposure -> 위험지수(Distance) -> 4단계 위험등급

    원본 HP 코드와 동일한 순서로 수행한다.
      1) 표준화 후 L2 노름으로 위험지수 산정
      2) 군집 중심 오름차순으로 군집번호를 재정렬(낮은 위험 -> 0)
      3) 위험지수를 군집 중심의 최소~최대 범위로 clip 한 뒤 재예측
      4) 군집번호를 등급명으로 변환
    """
    df[FEATURES] = df[FEATURES].fillna(0)
    scaled_features = scaler.transform(df[FEATURES])
    df['Distance'] = np.sqrt(np.sum(scaled_features ** 2, axis=1))
    df['Cluster'] = kmeans_model.predict(df[['Distance']])

    cluster_centers = kmeans_model.cluster_centers_.flatten()
    sorted_clusters = np.argsort(cluster_centers)
    label_map = {old: new for new, old in enumerate(sorted_clusters)}
    df['Cluster'] = df['Cluster'].map(label_map)

    min_distance = cluster_centers.min()
    max_distance = cluster_centers.max()
    df['Distance'] = np.clip(df['Distance'], min_distance, max_distance)
    df['Cluster'] = kmeans_model.predict(df[['Distance']])
    df['Cluster'] = df['Cluster'].map(label_map)

    df['Risk_Level'] = df['Cluster'].map(RISK_LEVEL_MAP)
    df['Grade'] = df['Cluster'].astype(int) + 1        # 1(매우낮음) ~ 4(매우위험)
    return df


# =============================================================================
# [G-1] 건물 위험등급
# =============================================================================
def run_risk_building(building_cur, models):
    """건물 객체별 침수위험등급 산정"""
    bd = building_cur.copy()

    bd['Hazard'] = bd['weight_word'] + bd['under_risk']
    bd['Vulnerability'] = bd['cont_vul']
    bd['Exposure'] = bd['alllbl']

    bd = compute_risk_grade(bd, models['bd_scaler'], models['bd_kmeans'])
    return bd


# =============================================================================
# [G-2] 도로 위험등급
# =============================================================================
def calculate_weight(row):
    """도로 노드 침수심 기반 손상가중치 (원본 HP_도로 와 동일)"""
    d = row['depth']
    if d == 0.0:
        return 0
    elif 0 < d < 0.5:
        return 0.5 * d
    elif 0.5 <= d < 1.0:
        return 0.34 * d + 0.08
    elif 1.0 <= d < 1.5:
        return 0.26 * d + 0.16
    elif 1.5 <= d < 2.0:
        return 0.2 * d + 0.25
    elif 2.0 <= d < 3.0:
        return 0.15 * d + 0.35
    elif 3.0 <= d < 4.0:
        return 0.1 * d + 0.5
    elif 4.0 <= d < 5.0:
        return 0.1 * d + 0.5
    else:
        return 1


def run_risk_road(road, models):
    """도로 노드별 침수위험등급 및 도로 손상도 산정"""
    rd = road.copy()

    rd['WEIGHT'] = rd.apply(calculate_weight, axis=1)
    rd['ROAD_DAMAGE_max'] = rd['LANES'] * 1 * 1
    rd['ROAD_DAMAGE'] = rd['LANES'] * 1 * rd['WEIGHT']

    rd['Hazard'] = rd['WEIGHT']
    rd['Vulnerability'] = rd['TI']
    rd['Exposure'] = rd['LANES']

    rd = compute_risk_grade(rd, models['rd_scaler'], models['rd_kmeans'])
    return rd


# =============================================================================
# [G-5] 행정동별 위험등급 분포
# =============================================================================
def aggregate_region_grade(bd_risk, rd_risk):
    """행정동별 건물·도로 위험등급 분포를 집계"""
    cols = ['매우위험', '위험', '낮음', '매우낮음']

    bd_counts = bd_risk.groupby(['ADM_CD', 'Risk_Level']).size().unstack(fill_value=0)
    for c in cols:
        if c not in bd_counts.columns:
            bd_counts[c] = 0
    bd_counts = bd_counts[cols].reset_index()
    bd_counts['ADM_CD'] = bd_counts['ADM_CD'].astype(str).str.split('.').str[0]

    rd_counts = rd_risk.groupby(['ADM_CD', 'Risk_Level']).size().unstack(fill_value=0)
    for c in cols:
        if c not in rd_counts.columns:
            rd_counts[c] = 0
    rd_counts = rd_counts[cols].reset_index()
    rd_counts.columns = ['ADM_CD'] + [f'도로_{c}' for c in cols]
    rd_counts['ADM_CD'] = rd_counts['ADM_CD'].astype(str).str.split('.').str[0]

    region = pd.merge(bd_counts, rd_counts, on='ADM_CD', how='outer').fillna(0)

    nm = (bd_risk[['ADM_CD', 'ADM_NM']].drop_duplicates()
          .assign(ADM_CD=lambda d: d['ADM_CD'].astype(str).str.split('.').str[0]))
    region = pd.merge(region, nm, on='ADM_CD', how='left')

    front = ['ADM_CD', 'ADM_NM']
    region = region[front + [c for c in region.columns if c not in front]]
    return region


# =============================================================================
# 침수위험등급 1스텝 수행
# =============================================================================
def run_risk(common_out, models, out_dir=None, step_name=None):
    """공통 전처리 결과로부터 침수위험등급(건물·도로·행정동)을 산정한다.

    Parameters
    ----------
    common_out : hp_common.run_common() 의 반환값
    models     : load_risk_models() 의 반환값
    out_dir    : 결과 저장 폴더 (None 이면 저장하지 않음)
    step_name  : 결과 파일 접두어 (예: 'H_120MIN')

    Returns
    -------
    dict : {'building_risk', 'road_risk', 'region_grade'}
    """
    bd_risk = run_risk_building(common_out['building_cur'], models)
    rd_risk = run_risk_road(common_out['road'], models)
    region_grade = aggregate_region_grade(bd_risk, rd_risk)

    result = {'building_risk': bd_risk, 'road_risk': rd_risk, 'region_grade': region_grade}

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pre = f"{step_name}_" if step_name else ""
        # ver 2.0 과 동일한 파일명으로 저장 (results_building_risk 는 ver 2.0 열 순서 + Grade)
        bd_risk.to_csv(os.path.join(out_dir, f"{pre}results_building_risk.csv"),
                       index=False, encoding='cp949')
        rd_risk.to_csv(os.path.join(out_dir, f"{pre}results_road_risk.csv"),
                       index=False, encoding='cp949')
        region_grade.to_csv(os.path.join(out_dir, f"{pre}region_risk_grade.csv"),
                            index=False, encoding='cp949')

    return result


# =============================================================================
# 단독 실행 (침수위험등급 단계만 확인할 때)
# =============================================================================
def main(asc_path, out_dir=None, step_name=None):
    """침수위험등급 단독 구동 : ASC 1개 -> 공통 전처리 -> 위험등급 결과파일"""
    import hp_common as common
    S = common.load_static_data()
    co = common.run_common(S, asc_path)
    models = load_risk_models()
    return run_risk(co, models, out_dir=out_dir, step_name=step_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python hp_risk.py <ASC 파일경로> [출력폴더]")
        sys.exit(1)
    asc = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else cfg.ensure_output_dir("RISK")
    name = os.path.basename(asc).replace('.asc', '')
    r = main(asc, out_dir=outd, step_name=name)
    print("[위험등급] 건물 %d개 / 도로노드 %d개 / 행정동 %d개 위험등급 산정 완료"
          % (len(r['building_risk']), len(r['road_risk']), len(r['region_grade'])))
