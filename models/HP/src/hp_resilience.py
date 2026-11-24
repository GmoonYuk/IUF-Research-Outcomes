# -*- coding: utf-8 -*-
"""
HP 모형 ver 3.0 — 3단계 : 회복력 · 복구우선순위 산정 (hp_resilience)
====================================================================
침수위험등급 결과를 입력으로 행정동별 지역 회복력(Resilience)과 건물별
복구우선순위(Restore_Priority)를 산정한다.
(ver 2.0 HP_건물_V3.ipynb · HP_도로.ipynb · HP_timeseries.ipynb 의 회복력·복구우선순위 산정부)

기능
  [R-1] 건물 견고성(Robustness-건물) 산정
        BI_i = (최대시점 피해액 합 - 현재시점 피해액 합) / 최대시점 피해액 합
  [R-2] 예측이재민 산정(RF + 다항회귀 병용) 및 자원부존성 산정
        자원부존성_i = 1 - 예측이재민_i / 대피소 수용가능인원_i
  [R-3] 도로망 실시간 기능 분석
        RA_i = 침수 후 0.172시간 이내 접근 경로 수 / 침수 전 경로 수   (신속성)
        SR_i = 침수 후 경계 내 총 최단경로 수 / 침수 전 경로 수        (중복성)
        RI_i = (최대 도로손상 - 현재 도로손상) / 최대 도로손상          (도로 견고성)
        + 우회 경로 기반 교통혼잡비용(TCC) 산정
  [R-4] 행정동별 지역 회복력(Resilience) 산정
        Robustness      = mean(견고성(건물), RI)
        Rapidness       = RA
        Redundancy      = SR
        Resourcefulness = 자원부존성
        Resilience      = (Robustness + Rapidness + Redundancy + Resourcefulness) / 4
  [R-5] 건물별 복구우선순위 산정
        Restore_Priority = Resilience x (1 - Distance / sqrt(3))
        ※ Distance 는 2단계(hp_risk)가 산정한 침수위험지수

입력
  - 공통 전처리 결과 (building_cur, building_max, road)
  - 침수위험등급 결과 (building_risk, road_risk)  ← 위험지수·위험등급·도로손상

출력 (시간단계별 폴더, 파일명·열 순서는 ver 2.0 과 동일)
  - H_(t)MIN_results_building.csv     : 행정동별 견고성(건물)·자원부존성·예측이재민·피해액
  - H_(t)MIN_results_road.csv         : 행정동별 RA·SR·RI·교통혼잡비용
  - H_(t)MIN_region_results_final.csv : 행정동별 회복력 4요소 및 Resilience
  - H_(t)MIN_building_risk_final.csv  : 건물별 복구우선순위 (건물 기초자료 순서)
  - H_(t)MIN_Priority.csv             : 건물별 복구우선순위 (순위 오름차순 정렬)

계산식과 상수는 ver 2.0 과 동일하며 변경하지 않았다.
ver 3.0 에서는 도로망 기준값(침수 전 최단경로 등)을 회차당 1회만 산정·캐싱하고,
RA 산정 시 출발노드별 최단경로를 1회만 계산하도록 구현을 바꾸었다 (산출값 동일).
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from joblib import Parallel, delayed
import warnings

import hp_config as cfg
from hp_common import safe_div

warnings.filterwarnings('ignore')

RISK_COLS = ['매우위험', '위험', '낮음', '매우낮음']


# =============================================================================
# 학습모델 적재
# =============================================================================
def load_resilience_models():
    """예측이재민 RF 모형, 회귀 스케일러, 회귀식을 적재한다."""
    return {
        'rf': joblib.load(cfg.P_RF_VICTIMS),
        'reg_scaler': joblib.load(cfg.P_REG_SCALER),
        'reg_equation': joblib.load(cfg.P_REG_EQUATION),
    }


def _predict_using_equation(X_scaled, equation_df, feature_names):
    """다항회귀식 적용 (원본과 동일)"""
    intercept = equation_df[equation_df['Term'] == 'Intercept']['Coefficient'].values[0]
    prediction = intercept
    for _, row in equation_df.iterrows():
        if row['Term'] != 'Intercept':
            term = row['Term']
            if term in feature_names:
                index = np.where(feature_names == term)[0][0]
                prediction += row['Coefficient'] * X_scaled[index]
    return prediction


# =============================================================================
# [R-1] ~ [R-2] 건물 회복력 요소
# =============================================================================
def run_resilience_building(building_cur, building_max, building_risk, S, models):
    """건물 견고성, 예측이재민, 자원부존성 산정"""
    building_depth = building_risk          # 현재 시점 건물 테이블 + 위험지수·위험등급
    building_depth_max = building_max

    # ---- [R-1] 행정동별 견고성(건물) -------------------------------------
    results = []
    grouped_restore = building_depth.groupby('ADM_CD')
    grouped_max = building_depth_max.groupby('ADM_CD')
    used_ids = {}

    for group, df_restore_group in grouped_restore:
        ids_list = df_restore_group['id'].tolist()
        used_ids[group] = ids_list
        total_f_total_damage = df_restore_group['f_total_damage'].sum()

        if group in grouped_max.groups:
            df_max_group = grouped_max.get_group(group)
            df_max_group_filtered = df_max_group[df_max_group['id'].isin(used_ids[group])]
            total_max_damage = df_max_group_filtered['f_total_damage'].sum()

            overall_BI_group = safe_div((total_max_damage - total_f_total_damage), total_max_damage)
            adm_nm = df_restore_group['ADM_NM'].iloc[0]
            results.append([group, adm_nm, overall_BI_group])

    results_df = pd.DataFrame(results, columns=['ADM_CD', 'ADM_NM', '견고성(건물)'])

    # ---- 위험등급 분포 결합 ----------------------------------------------
    risk_counts = building_depth.groupby(['ADM_CD', 'Risk_Level']).size().unstack(fill_value=0)
    results_df = results_df.merge(risk_counts, on='ADM_CD', how='left')
    for col in RISK_COLS:
        if col not in results_df.columns:
            results_df[col] = 0
        else:
            results_df[col] = results_df[col].fillna(0)

    # ---- [R-2] 예측이재민 -------------------------------------------------
    victims = S['victims_input'].copy()

    filtered = building_depth[building_depth['depth'] > cfg.D_PASSABLE]
    bd_grouped = filtered.groupby('ADM_CD').size().reset_index(name='침수건물')

    victims = victims.merge(bd_grouped, on='ADM_CD', how='left', suffixes=('', '_new'))
    victims['침수건물'] = victims['침수건물_new'].fillna(0)
    victims = victims.drop(columns=['침수건물_new'])

    victims['재난복구지원금 (천원)'] = victims['침수건물'] * safe_div(cfg.RECOVERY_FUND_PER_BD, 1000.0)

    total_damage_sum = filtered.groupby('ADM_CD')['f_cont_damage'].sum().reset_index(name='피해액 (천원)')
    victims = victims.merge(total_damage_sum, on='ADM_CD', how='left', suffixes=('', '_new'))
    victims['피해액 (천원)'] = victims['피해액 (천원)_new'].fillna(0)
    victims = victims.drop(columns=['피해액 (천원)_new'])

    housing = filtered[filtered['bdtyp'].isin(['다세대주택/연립주택', '단독주택', '아파트'])]
    housing_counts = housing.groupby('ADM_CD').size().reset_index(name='주택침수세대 (세대)')
    victims = victims.merge(housing_counts, on='ADM_CD', how='left', suffixes=('', '_new'))
    victims['주택침수세대 (세대)'] = victims['주택침수세대 (세대)_new'].fillna(0)
    victims = victims.drop(columns=['주택침수세대 (세대)_new'])

    housing_damage = housing.groupby('ADM_CD')['f_cont_damage'].sum().reset_index(name='주택피해액 (천원)')
    victims = victims.merge(housing_damage, on='ADM_CD', how='left', suffixes=('', '_new'))
    victims['주택피해액 (천원)'] = victims['주택피해액 (천원)_new'].fillna(0)
    victims = victims.drop(columns=['주택피해액 (천원)_new'])

    underground = filtered[(filtered['bdtyp'].isin(['다세대주택/연립주택', '단독주택', '아파트'])) &
                           (filtered['und_flo_cd'] != 0)]
    under_counts = underground.groupby('ADM_CD').size().reset_index(name='(반)지하주택침수세대 (세대)')
    victims = victims.merge(under_counts, on='ADM_CD', how='left', suffixes=('', '_new'))
    victims['(반)지하주택침수세대 (세대)'] = victims['(반)지하주택침수세대 (세대)_new'].fillna(0)
    victims = victims.drop(columns=['(반)지하주택침수세대 (세대)_new'])

    input_features = ['100', '200', '300', '400', '500', '600', '700', '평균고도', '평균경사',
                      '침수건물', '주택침수세대 (세대)', '주택피해액 (천원)', '피해액 (천원)',
                      '(반)지하주택침수세대 (세대)', '면적', '인구수(명)']
    X_input = victims[input_features].fillna(0)
    X_input_scaled = models['reg_scaler'].transform(X_input)

    predictions = []
    for index, row in victims.iterrows():
        rf_prediction = models['rf'].predict([X_input_scaled[index]])[0]
        if (rf_prediction >= cfg.VICTIM_RF_TH
                or row['침수건물'] >= cfg.VICTIM_BLDG_TH
                or row['주택침수세대 (세대)'] >= cfg.VICTIM_HOUSE_TH
                or row['(반)지하주택침수세대 (세대)'] >= cfg.VICTIM_UNDER_TH):
            poly_prediction = _predict_using_equation(X_input_scaled[index],
                                                      models['reg_equation'], X_input.columns)
            if poly_prediction >= cfg.VICTIM_POLY_MAX or poly_prediction < 0:
                predictions.append(rf_prediction)
            else:
                predictions.append(poly_prediction)
        else:
            predictions.append(rf_prediction)

    victims['예측이재민'] = predictions

    for col in ['예측이재민', '피해액 (천원)', '침수건물', '재난복구지원금 (천원)']:
        results_df = results_df.merge(victims[['ADM_CD', col]], on='ADM_CD', how='left')

    # ---- 자원부존성 --------------------------------------------------------
    results_df = results_df.merge(S['shelter_adm'][['ADM_CD', '수용가능인원수']],
                                  on='ADM_CD', how='left')
    results_df['자원부존성'] = 0.0
    for idx, row in results_df.iterrows():
        results_df.at[idx, '자원부존성'] = 1 - safe_div(row['예측이재민'], row['수용가능인원수'])
    results_df['자원부존성'] = results_df['자원부존성'].fillna(0.0).clip(lower=0.0)
    results_df = results_df.drop(columns=['수용가능인원수'])

    return results_df[['ADM_CD', 'ADM_NM', '피해액 (천원)', '재난복구지원금 (천원)', '예측이재민',
                       '견고성(건물)', '자원부존성'] + RISK_COLS]


# =============================================================================
# [R-3] 도로망 실시간 기능 분석
# =============================================================================
def _process_ra_source_node(start_node, dest_set, G, t_limit):
    if not G.has_node(start_node):
        return 0
    try:
        lengths, paths = nx.single_source_dijkstra(G, start_node, weight='LENGTH')
    except Exception:
        return 0

    count = 0
    valid_targets = dest_set.intersection(paths.keys())
    for target in valid_targets:
        if start_node == target:
            continue
        length = lengths[target]
        if length == 0:
            continue
        path = paths[target]
        path_edges = list(zip(path[:-1], path[1:]))
        weighted_sum = sum(G[u][v].get('SPD_LEN', 0) for u, v in path_edges)
        path_average_speed = weighted_sum / length if length > 0 else 1
        if (length / 1000) / path_average_speed <= t_limit:
            count += 1
    return count


def _process_sr_source_node(start_node, dest_set, G, group):
    if not G.has_node(start_node):
        return []
    try:
        lengths, paths = nx.single_source_dijkstra(G, start_node, weight='LENGTH')
    except Exception:
        return []

    path_infos = []
    valid_targets = dest_set.intersection(paths.keys())
    for target in valid_targets:
        if start_node == target:
            continue
        path = paths[target]
        length = lengths[target]
        path_edges = list(zip(path[:-1], path[1:]))
        total_time = sum(G[u][v].get('TIME', 0) for u, v in path_edges)
        total_amount = sum(G[u][v].get('설계교통량', 0) for u, v in path_edges)
        avg_amount = total_amount / len(path_edges) if path_edges else 0
        if total_time > 0:
            path_infos.append({
                'Group': group,
                'Path_Name': f"{start_node},{target}",
                'Path': ','.join(map(str, path)),
                'Length': length / 1000,
                'Time': total_time,
                'Speed_ave': (length / 1000) / total_time,
                'Average_Amount': avg_amount,
            })
    return path_infos


def _tcc_columns(df):
    """차종별 교통혼잡비용 산정 (원본과 동일)"""
    sa, sb = df['Speed_ave_after'], df['Speed_ave_before']
    dt = (df['Time_after'] - df['Time_before']) / 3600
    L = df['Length_after']
    A = df['Average_Amount_after']

    df['car_TCC'] = A * 0.83 * (402.46 * L * (
        (0.02882 + 0.91 / sa + 0.000003828 * sa ** 2)
        - (0.02882 + 0.91 / sb + 0.000003828 * sb ** 2)) + 23289 * dt)
    df['bus_TCC'] = A * 0.03 * (431.85 * L * (
        (0.02906 + 2.3225 / sa + (8.541 * 10 ** (-6)) * sa ** 2)
        - (0.02906 + 2.3225 / sb + (8.541 * 10 ** (-6)) * sb ** 2)) + (25616 + 105674) * dt)
    df['truck_TCC'] = A * 0.14 * (431.85 * L * (
        (0.04167 + 2.725 / sa + (2.086 * 10 ** (-5)) * sa ** 2)
        - (0.04167 + 2.725 / sb + (2.086 * 10 ** (-5)) * sb ** 2)) + 28888 * dt)
    return df



def _chunks(seq, n_parts):
    """병렬 태스크당 여러 출발노드를 묶어 그래프 직렬화 비용을 분산"""
    seq = list(seq)
    if not seq:
        return []
    n_parts = max(1, min(n_parts, len(seq)))
    size = -(-len(seq) // n_parts)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _n_parts(n_jobs):
    import os as _os
    cores = (_os.cpu_count() or 1) if n_jobs in (-1, None) else max(1, int(n_jobs))
    return cores * 4


def _ra_source_group_counts(start_node, dest_list, universe, G, t_limit):
    """출발노드 1개에서 최단경로를 1회만 계산하고 그룹별 조건 충족 목적지 수를 반환.

    원본은 행정동 그룹마다 동일한 출발노드의 Dijkstra 를 다시 수행했으나,
    출발노드가 같으면 최단경로 결과가 같으므로 1회만 계산하고
    그룹별 목적지 집합과 교집합을 취한다. 산출값은 원본과 동일하다.
    """
    ng = len(dest_list)
    if not G.has_node(start_node):
        return [0] * ng
    try:
        lengths, paths = nx.single_source_dijkstra(G, start_node, weight='LENGTH')
    except Exception:
        return [0] * ng

    ok = set()
    for target in universe.intersection(paths.keys()):
        if start_node == target:
            continue
        length = lengths[target]
        if length == 0:
            continue
        path = paths[target]
        path_edges = list(zip(path[:-1], path[1:]))
        weighted_sum = sum(G[u][v].get('SPD_LEN', 0) for u, v in path_edges)
        path_average_speed = weighted_sum / length if length > 0 else 1
        if (length / 1000) / path_average_speed <= t_limit:
            ok.add(target)
    return [len(ok & ds) for ds in dest_list]


def _ra_batch(start_nodes, dest_list, universe, G, t_limit):
    tot = [0] * len(dest_list)
    for sn in start_nodes:
        row = _ra_source_group_counts(sn, dest_list, universe, G, t_limit)
        for i, v in enumerate(row):
            tot[i] += v
    return tot


def _ra_group_counts(G, start_nodes, groups, dest_sets, t_limit, n_jobs):
    """행정동별 '통행시간 기준 이내 최단경로 수' 를 한 번의 순회로 산정"""
    dest_list = [dest_sets[g] for g in groups]
    universe = set().union(*dest_list) if dest_list else set()
    batches = _chunks(start_nodes, _n_parts(n_jobs))
    if not batches:
        return {g: 0 for g in groups}
    res = Parallel(n_jobs=n_jobs)(
        delayed(_ra_batch)(b, dest_list, universe, G, t_limit) for b in batches)
    tot = [0] * len(groups)
    for row in res:
        for i, v in enumerate(row):
            tot[i] += v
    return {g: tot[i] for i, g in enumerate(groups)}


def _sr_batch(start_nodes, dest_set, G, group):
    out = []
    for sn in start_nodes:
        out.extend(_process_sr_source_node(sn, dest_set, G, group))
    return out


def _sr_paths(G, start_nodes, dest_set, group, n_jobs):
    """경계 내 최단경로 목록 산정 (출발노드를 배치로 묶어 병렬 수행)"""
    batches = _chunks(start_nodes, _n_parts(n_jobs))
    if not batches:
        return []
    res = Parallel(n_jobs=n_jobs)(
        delayed(_sr_batch)(b, dest_set, G, group) for b in batches)
    return [i for sub in res for i in sub]


# =============================================================================
# 도로망 기준값 캐시  (침수 여부와 무관 -> 회차당 1회만 산정)
# =============================================================================
def build_road_baseline(S, n_jobs=-1):
    """침수 여부와 무관한 도로망 기준값을 1회 산정하여 캐싱한다.

    아래 값들은 도로망 링크자료(link_df)와 노드의 행정동 코드에만 의존하므로
    10분 단위 시간단계가 바뀌어도 값이 변하지 않는다.

      · 인접 행정구역 목록 및 주요 노드 / 경계 노드 집합
      · RA 그래프 G_RA (통행시간 판정용)와 SR 그래프 G_SR (중복성 판정용)
      · 침수 전 기준 경로 : 0.172시간 이내 총 최단경로 수, 경계 내 총 최단경로 수·길이·시간
      · 행정동별 최대 도로손상량 ROAD_DAMAGE_max 합계

    원본 HP_도로 코드는 이 값들을 매 시간단계마다 다시 계산했으나, 값이 동일하므로
    회차 시작 시 1회만 산정하여 재사용한다. 산출 결과값은 원본과 완전히 같다.
    본 산정 시간은 기초자료 적재와 마찬가지로 시간단계별 구동시간에 포함하지 않고
    별도로 측정하여 기록한다.
    """
    link_df = S['link_df']
    sig_df = S['sig_df']
    main_df = S['main_df'].copy()
    boundary_all_node_df = S['all_node_df'].copy()
    no_node = S['no_node'][['NODE_ID']].iloc[0:0]

    base = S['df_rd_base'].copy()

    # ---- 인접 행정구역 추출 ------------------------------------------------
    base['ADM_CD'] = base['ADM_CD'].astype(str)
    base['ADM_CD_5'] = base['ADM_CD'].str[:5]
    unique_adm_cd = base['ADM_CD_5'].unique()

    def get_adjacent_adm(adm_code, df):
        return df.loc[df[adm_code] == 1, 'ADM_CD'].tolist()

    adjacent_list = [get_adjacent_adm(a, sig_df) for a in unique_adm_cd]
    flat_list = list(set([i for sub in adjacent_list for i in sub]))

    main_df['ADM_CD_5'] = main_df['ADM_CD'].astype(str).str[:5].astype(int)
    boundary_all_node_df['ADM_CD_5'] = boundary_all_node_df['ADM_CD'].astype(str).str[:5].astype(int)
    main_df = main_df[main_df['ADM_CD_5'].isin(flat_list)]
    boundary_all_node_df = boundary_all_node_df[boundary_all_node_df['ADM_CD_5'].isin(flat_list)]
    main_df.drop(columns=['ADM_CD_5'], inplace=True)
    boundary_all_node_df.drop(columns=['ADM_CD_5'], inplace=True)

    boundary_node_id = base['NODE_ID'].tolist()
    combined_ids_with_adjacent_node = boundary_all_node_df['NODE_ID']

    # ---- RA 그래프 ---------------------------------------------------------
    link_df_RA = link_df[link_df['F_NODE'].isin(combined_ids_with_adjacent_node)
                         | link_df['T_NODE'].isin(combined_ids_with_adjacent_node)]
    G_RA = nx.from_pandas_edgelist(link_df_RA, source='F_NODE', target='T_NODE',
                                   edge_attr=['LENGTH', 'MAX_SPD'])
    for u, v, d in G_RA.edges(data=True):
        d['SPD_LEN'] = d.get('MAX_SPD', 1) * d.get('LENGTH', 0)

    # ---- SR 그래프 ---------------------------------------------------------
    link_df_SR = link_df[(link_df['F_NODE'].isin(boundary_node_id))
                         | (link_df['T_NODE'].isin(boundary_node_id))]
    G_SR = nx.from_pandas_edgelist(link_df_SR, source='F_NODE', target='T_NODE',
                                   edge_attr=['LENGTH', 'MAX_SPD', '설계교통량'])
    for u, v, d in G_SR.edges(data=True):
        d['TIME'] = (d.get('LENGTH', 0) / 1000) / d.get('MAX_SPD', 1) if d.get('MAX_SPD', 1) > 0 else float('inf')

    # ---- 그룹·노드 집합 ----------------------------------------------------
    valid_nodes = set(no_node['NODE_ID'].unique())
    groups = list(base.groupby('ADM_NM').groups.keys())
    start_nodes_RA = main_df['NODE_ID'].unique()
    dest_sets = {g: set(base[base['ADM_NM'] == g]['NODE_ID'].unique()) for g in groups}
    sr_nodes = {g: base[(base['ADM_NM'] == g)
                        & ~(base['NODE_ID'].isin(valid_nodes))]['NODE_ID'].unique()
                for g in groups}

    tlim = cfg.TRAVEL_TIME_LIMIT

    # ---- 침수 전 RA 기준 경로 ----------------------------------------------
    group_0172_before = _ra_group_counts(G_RA, start_nodes_RA, groups, dest_sets, tlim, n_jobs)

    # ---- 침수 전 SR 기준 경로 ----------------------------------------------
    sp_before, sp_len_before, sp_time_before = {}, {}, {}
    all_paths = []
    for g in groups:
        ns = sr_nodes[g]
        flat = _sr_paths(G_SR, ns, set(ns), g, n_jobs)
        sp_before[g] = len(flat)
        sp_len_before[g] = sum(p['Length'] for p in flat)
        sp_time_before[g] = sum(p['Time'] for p in flat)
        all_paths.extend(flat)
    paths_df_before = pd.DataFrame(all_paths)

    # ---- 행정동별 최대 도로손상량 ------------------------------------------
    road_damage_max = (base['LANES'] * 1 * 1).groupby(base['ADM_NM']).sum()

    return {
        'main_df': main_df,
        'G_RA': G_RA, 'G_SR': G_SR,
        'groups': groups,
        'start_nodes_RA': start_nodes_RA,
        'dest_sets': dest_sets,
        'sr_nodes': sr_nodes,
        'valid_nodes': valid_nodes,
        'group_0172_before': group_0172_before,
        'sp_before': sp_before,
        'sp_len_before': sp_len_before,
        'sp_time_before': sp_time_before,
        'paths_df_before': paths_df_before,
        'road_damage_max': road_damage_max,
    }


def get_road_baseline(S, n_jobs=-1):
    """캐시된 도로망 기준값을 반환한다. 캐시가 없으면 즉시 산정한다."""
    if '_road_baseline' not in S:
        S['_road_baseline'] = build_road_baseline(S, n_jobs=n_jobs)
    return S['_road_baseline']


def run_resilience_road(road, road_risk, S, n_jobs=-1):
    """도로망 RA · SR · RI · TCC 산정

    침수 전 기준 경로는 build_road_baseline() 이 산정한 캐시를 사용하고,
    침수를 반영한 경로만 시간단계마다 새로 산정한다.
    """
    B = get_road_baseline(S, n_jobs=n_jobs)

    TCC_0 = S['tcc_df'].copy()

    depth_df = road.copy()
    depth_2_df = road.copy()
    boundary_node_df = road_risk.copy()          # 도로 노드 위험등급 결과 (WEIGHT, ROAD_DAMAGE 포함)

    depth_df_boundary = depth_df[depth_df['depth'] > cfg.D_PASSABLE]
    destination_df = depth_df

    main_df = B['main_df']
    G_RA, G_SR = B['G_RA'], B['G_SR']
    groups = B['groups']
    valid_nodes = B['valid_nodes']
    tlim = cfg.TRAVEL_TIME_LIMIT

    grouped_df = pd.DataFrame(list(B['group_0172_before'].items()),
                              columns=['ADM_NM', '0.172 이내 총 최단 경로 수'])
    sp_before = B['sp_before']
    sp_len_before = B['sp_len_before']
    sp_time_before = B['sp_time_before']
    paths_df_before = B['paths_df_before']

    # ---- 침수 반영 RA ------------------------------------------------------
    G_RA_prime = G_RA.copy()
    flooded_nodes = destination_df[destination_df['depth'] > cfg.D_PASSABLE]['NODE_ID'].tolist()
    G_RA_prime.remove_nodes_from([n for n in flooded_nodes if G_RA_prime.has_node(n)])

    group_0172_flood = _ra_group_counts(G_RA_prime, B['start_nodes_RA'], groups,
                                        B['dest_sets'], tlim, n_jobs)
    grouped_df['0.172 이내 총 최단 경로 수_flood'] = grouped_df['ADM_NM'].map(group_0172_flood)

    # ---- 침수 반영 SR ------------------------------------------------------
    depth_df_filtered = depth_2_df[depth_2_df['depth'] > cfg.D_PASSABLE]
    depth_df_filtered_ids = depth_df_filtered['NODE_ID'].tolist()
    all_boundary_node_ids = boundary_node_df['NODE_ID'].unique().tolist()
    depth0_node_ids = set([n for n in all_boundary_node_ids if n not in depth_df_filtered_ids])

    G_SR_prime = G_SR.copy()
    depth_dict = depth_df.set_index('NODE_ID')['depth'].to_dict()
    final_nodes_to_remove = []
    for node in depth_df_filtered_ids:
        if G_SR_prime.has_node(node):
            neighbors = list(G_SR_prime.neighbors(node))
            if all(depth_dict.get(nb, -1) > 0 for nb in neighbors):
                final_nodes_to_remove.append(node)
    G_SR_prime.remove_nodes_from(final_nodes_to_remove)

    sp_after, sp_len_after, sp_time_after = {}, {}, {}
    all_paths_after = []
    for group in groups:
        start_nodes = B['sr_nodes'][group]
        dest_set = set(boundary_node_df[(boundary_node_df['ADM_NM'] == group)
                                        & (boundary_node_df['NODE_ID'].isin(depth0_node_ids))
                                        & ~(boundary_node_df['NODE_ID'].isin(valid_nodes))]['NODE_ID'].unique())
        flat = _sr_paths(G_SR_prime, start_nodes, dest_set, group, n_jobs)
        for p in flat:
            p['Time'] = p['Time'] * 2
            p['Speed_ave'] = p['Length'] / p['Time'] if p['Time'] > 0 else 0
        sp_after[group] = len(flat)
        sp_len_after[group] = sum(p['Length'] for p in flat)
        sp_time_after[group] = sum(p['Time'] for p in flat)
        all_paths_after.extend(flat)

    paths_df_after = pd.DataFrame(all_paths_after)

    grouped_df['총_최단_경로_수_flood'] = grouped_df['ADM_NM'].map(sp_after)
    grouped_df['총_최단_경로_길이_flood'] = grouped_df['ADM_NM'].map(sp_len_after)
    grouped_df['총_최단_경로_시간_flood'] = grouped_df['ADM_NM'].map(sp_time_after)
    grouped_df['총_최단_경로_수'] = grouped_df['ADM_NM'].map(sp_before)
    grouped_df['총_최단_경로_길이'] = grouped_df['ADM_NM'].map(sp_len_before)
    grouped_df['총_최단_경로_시간'] = grouped_df['ADM_NM'].map(sp_time_before)

    # ---- 행정동별 도로 위험등급 분포 결합 ----------------------------------
    risk_counts = boundary_node_df.groupby(['ADM_CD', 'ADM_NM', 'Risk_Level']).size().unstack(fill_value=0)
    results_road_df = risk_counts.reset_index()[['ADM_CD', 'ADM_NM']]
    for col in ['매우낮음', '낮음', '위험', '매우위험']:
        results_road_df[col] = risk_counts[col].values if col in risk_counts.columns else 0

    merged_df = pd.merge(results_road_df, grouped_df, on='ADM_NM', how='left')

    # ---- 교통혼잡비용(TCC) -------------------------------------------------
    node_ids = (list(set(depth_df['NODE_ID']) - set(depth_2_df['NODE_ID']))
                if set(depth_df['NODE_ID']) != set(depth_2_df['NODE_ID']) else ['0'])
    node_ids = [str(n) for n in node_ids]

    grouped_tcc_sums = pd.DataFrame(columns=['ADM_NM', 'car_TCC', 'bus_TCC', 'truck_TCC'])

    if not paths_df_before.empty and not paths_df_after.empty:
        mdf = pd.merge(paths_df_before, paths_df_after, on='Path_Name',
                       suffixes=('_before', '_after'))
        adm_nm_set = set(depth_df_boundary['ADM_NM'])

        def filter_path(row):
            if '0' in node_ids:
                return (row['Path_before'] != row['Path_after'], False)
            nodes_after = str(row['Path_after']).split(',')[0:]
            condition_after = not any(n in node_ids for n in nodes_after)
            condition_path_diff = row['Path_before'] != row['Path_after']
            return (condition_path_diff, condition_after and condition_path_diff)

        flags = mdf.apply(filter_path, axis=1)
        condition_after_df = mdf[flags.apply(lambda x: x[1])]
        bypassed = mdf[(mdf['Length_after'] > mdf['Length_before']) & flags.apply(lambda x: x[0])]

        mdf_f = mdf[(mdf['Length_after'] == mdf['Length_before'])
                    & ~mdf['Path_Name'].isin(bypassed['Path_Name'])]
        mdf_f = mdf_f[~mdf_f['Group_after'].isin(adm_nm_set)]
        final_merged_df = pd.concat([condition_after_df, mdf_f], ignore_index=True).drop_duplicates()

        if not bypassed.empty and not final_merged_df.empty:
            bypassed = bypassed.copy()
            final_merged_df = final_merged_df.copy()
            bypassed['Start_Node'] = bypassed['Path_Name'].str.split(',').str[0]
            final_merged_df['Start_Node'] = final_merged_df['Path_Name'].str.split(',').str[0]

            idx1 = final_merged_df.groupby('Start_Node')['Length_after'].idxmax()
            tcc_paths = final_merged_df.loc[idx1].copy()
            idx2 = bypassed.groupby('Start_Node')['Length_after'].idxmax()
            bypassed = bypassed.loc[idx2].copy()

            bypassed = _tcc_columns(bypassed)
            tcc_paths = _tcc_columns(tcc_paths)

            b_f = bypassed[(bypassed['car_TCC'] > 0) & (bypassed['bus_TCC'] > 0) & (bypassed['truck_TCC'] > 0)]
            t_f = tcc_paths[(tcc_paths['car_TCC'] > 0) & (tcc_paths['bus_TCC'] > 0) & (tcc_paths['truck_TCC'] > 0)]

            combined = pd.concat([b_f, t_f], ignore_index=True)
            combined = combined.drop_duplicates(subset='Path_Name', keep='first', ignore_index=True)

            counts = combined['Path_Name'].value_counts().to_dict()
            TCC_0['count'] = TCC_0['Path_Name'].map(counts).fillna(0).astype(int)
            paths_to_remove = TCC_0[TCC_0['count'] == 0]['Path_Name'].tolist()
            combined = combined[~combined['Path_Name'].isin(paths_to_remove)]

            grouped_tcc_sums = combined.groupby('Group_before').agg(
                {'car_TCC': 'sum', 'bus_TCC': 'sum', 'truck_TCC': 'sum'}).reset_index()
            grouped_tcc_sums.rename(columns={'Group_before': 'ADM_NM'}, inplace=True)

    merged_df['car_TCC'] = 0.0
    merged_df['bus_TCC'] = 0.0
    merged_df['truck_TCC'] = 0.0
    merged_df = pd.merge(merged_df, grouped_tcc_sums, on='ADM_NM', how='left', suffixes=('', '_new'))
    for col in ['car_TCC_new', 'bus_TCC_new', 'truck_TCC_new']:
        if col in merged_df.columns:
            merged_df[col.replace('_new', '')] += merged_df[col].fillna(0)
            merged_df.drop(columns=[col], inplace=True)
    merged_df['TCC'] = merged_df['car_TCC'] + merged_df['bus_TCC'] + merged_df['truck_TCC']

    # ---- RA / SR / RI ------------------------------------------------------
    merged_df['RA'] = np.where(merged_df['0.172 이내 총 최단 경로 수'] > 0,
                               merged_df['0.172 이내 총 최단 경로 수_flood']
                               / merged_df['0.172 이내 총 최단 경로 수'], 1.0)
    merged_df['SR'] = np.where(merged_df['총_최단_경로_수'] > 0,
                               merged_df['총_최단_경로_수_flood']
                               / merged_df['총_최단_경로_수'], 1.0)

    ri_df = pd.DataFrame({'ADM_NM': list(groups)}).set_index('ADM_NM')
    ri_df['ROAD_DAMAGE_MAX'] = B['road_damage_max']
    ri_df['ROAD_DAMAGE'] = boundary_node_df.groupby('ADM_NM')['ROAD_DAMAGE'].sum()
    ri_df['RI'] = np.where(ri_df['ROAD_DAMAGE_MAX'] > 0,
                           (ri_df['ROAD_DAMAGE_MAX'] - ri_df['ROAD_DAMAGE']) / ri_df['ROAD_DAMAGE_MAX'], 1.0)

    merged_df = pd.merge(merged_df, ri_df['RI'], left_on='ADM_NM', right_index=True, how='left')
    if 'RI_y' in merged_df.columns:
        merged_df.rename(columns={'RI_y': 'RI'}, inplace=True)
    if 'RI_x' in merged_df.columns:
        merged_df.drop(columns=['RI_x'], inplace=True)

    merged_df['RA'] = merged_df['RA'].fillna(1.0).round(3)
    merged_df['SR'] = merged_df['SR'].fillna(1.0).round(3)
    merged_df['RI'] = merged_df['RI'].fillna(1.0).round(3)
    merged_df['TCC'] = merged_df['TCC'].fillna(0).apply(lambda x: f"{x:,.0f}")

    return merged_df


# =============================================================================
# [R-4] 지역 회복력 통합
# =============================================================================
def merge_resilience(building_results, road_results):
    """건물·도로 결과를 병합하여 행정동별 Resilience 산정 (원본과 동일)"""
    b = building_results.copy()
    r = road_results.copy()
    b['ADM_CD'] = b['ADM_CD'].astype(str).str.split('.').str[0]
    r['ADM_CD'] = r['ADM_CD'].astype(str).str.split('.').str[0]
    HP = pd.merge(b, r, on='ADM_CD', how='outer')

    if 'ADM_NM' not in HP.columns:
        if 'ADM_NM_x' in HP.columns:
            HP['ADM_NM'] = HP['ADM_NM_x']
        elif 'ADM_NM_y' in HP.columns:
            HP['ADM_NM'] = HP['ADM_NM_y']

    for col in ['견고성(건물)', 'RI', 'RA', 'SR', '자원부존성']:
        if col not in HP.columns:
            HP[col] = 1.0 if col != '자원부존성' else 0.0

    HP['Robustness'] = HP[['견고성(건물)', 'RI']].mean(axis=1)
    HP['Rapidness'] = HP['RA']
    HP['Redundancy'] = HP['SR']
    HP['Resourcefulness'] = HP['자원부존성']
    HP['Resilience'] = (HP['Robustness'] + HP['Rapidness']
                        + HP['Redundancy'] + HP['Resourcefulness']) / 4
    return HP


# =============================================================================
# [R-5] 복구우선순위
# =============================================================================
def restore_priority(building_risk, region_resilience):
    """건물별 복구우선순위 산정 (원본과 동일)

    Restore_Priority = Resilience x (1 - Distance / sqrt(3))
    """
    b = building_risk.copy()
    reg = region_resilience.copy()
    b['ADM_CD'] = b['ADM_CD'].astype(str).str.split('.').str[0]
    reg['ADM_CD'] = reg['ADM_CD'].astype(str).str.split('.').str[0]

    out = pd.merge(b, reg[['ADM_CD', 'Resilience']], on='ADM_CD', how='left')
    out['Restore_Priority'] = out['Resilience'] * (1 - out['Distance'] / (3 ** 0.5))
    out['Restore_Priority_Rank'] = out['Restore_Priority'].rank(method='dense',
                                                                ascending=True).astype('Int64')
    # ver 2.0 열 순서 유지 : ver 3.0 에서 추가된 Grade 열은 맨 뒤에 둔다
    if 'Grade' in out.columns:
        out = out[[c for c in out.columns if c != 'Grade'] + ['Grade']]
    # 행 순서 : 복구우선순위 순위 오름차순. 동순위 건물은 건물 기초자료 순서를 유지한다(안정 정렬).
    #   ※ ver 2.0 의 기본 정렬은 동순위 행의 순서가 실행 환경(CPU·라이브러리)에 따라 달라질 수 있었다.
    #   index 는 건물 기초자료 순서를 보존하므로 sort_index() 로 원래 순서를 복원할 수 있다.
    return out.sort_values(by='Restore_Priority_Rank', kind='stable')


# =============================================================================
# 회복력 · 복구우선순위 1스텝 수행
# =============================================================================
def run_resilience(common_out, risk_out, S, models, out_dir=None, step_name=None, n_jobs=-1):
    """침수위험등급 결과를 입력으로 회복력·복구우선순위를 산정한다.

    Returns
    -------
    dict : {'building_results', 'road_results', 'region_resilience', 'priority', 'timing'}
           timing 은 내부 구간별 소요시간(초)
             T_P : 입력 준비(위험등급 산출물 적재)
             T_B : 건물 회복력 구간(견고성·예측이재민·자원부존성)
             T_R : 도로망 구간(RA·SR·RI·교통혼잡비용)
             T_M : 병합·회복력 산정·복구우선순위 저장 구간
    """
    import time
    t0 = time.perf_counter()
    building_cur = common_out['building_cur']
    building_max = common_out['building_max']
    road = common_out['road']
    bd_risk = risk_out['building_risk']
    rd_risk = risk_out['road_risk']
    t1 = time.perf_counter()

    b_res = run_resilience_building(building_cur, building_max, bd_risk, S, models)
    t2 = time.perf_counter()

    r_res = run_resilience_road(road, rd_risk, S, n_jobs=n_jobs)
    t3 = time.perf_counter()

    region = merge_resilience(b_res, r_res)
    prio = restore_priority(bd_risk, region)

    result = {'building_results': b_res, 'road_results': r_res,
              'region_resilience': region, 'priority': prio,
              'timing': {'T_P': round(t1 - t0, 3), 'T_B': round(t2 - t1, 3),
                         'T_R': round(t3 - t2, 3), 'T_M': None}}

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pre = f"{step_name}_" if step_name else ""
        # ver 2.0 과 동일한 파일명·열 순서로 저장
        b_res.to_csv(os.path.join(out_dir, f"{pre}results_building.csv"),
                     index=False, encoding='cp949')
        r_res.to_csv(os.path.join(out_dir, f"{pre}results_road.csv"),
                     index=False, encoding='cp949')
        region.to_csv(os.path.join(out_dir, f"{pre}region_results_final.csv"),
                      index=False, encoding='cp949')
        # building_risk_final : 건물 기초자료 순서 / Priority : 복구우선순위 순위 오름차순
        prio.sort_index().to_csv(os.path.join(out_dir, f"{pre}building_risk_final.csv"),
                                 index=False, encoding='cp949')
        prio.to_csv(os.path.join(out_dir, f"{pre}Priority.csv"),
                    index=False, encoding='cp949')

    result['timing']['T_M'] = round(time.perf_counter() - t3, 3)
    return result


# =============================================================================
# 단독 실행 (회복력·복구우선순위 단계만 확인할 때)
# =============================================================================
def main(asc_path, out_dir=None, step_name=None):
    """단독 구동 : 공통 전처리 + 침수위험등급을 먼저 산정한 뒤 회복력·복구우선순위를 수행"""
    import hp_common as common
    import hp_risk as risk

    S = common.load_static_data()
    co = common.run_common(S, asc_path)
    risk_models = risk.load_risk_models()
    risk_out = risk.run_risk(co, risk_models)

    resilience_models = load_resilience_models()
    return run_resilience(co, risk_out, S, resilience_models, out_dir=out_dir, step_name=step_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python hp_resilience.py <ASC 파일경로> [출력폴더]")
        sys.exit(1)
    asc = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else cfg.ensure_output_dir("RESILIENCE")
    name = os.path.basename(asc).replace('.asc', '')
    r = main(asc, out_dir=outd, step_name=name)
    print("[회복력] 행정동 %d개 회복력 및 건물 %d개 복구우선순위 산정 완료"
          % (len(r['region_resilience']), len(r['priority'])))
