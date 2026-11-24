# -*- coding: utf-8 -*-
"""
HP 모형 ver 3.0 — 1단계 : 공통 전처리 (hp_common)
====================================================================
침수위험등급(hp_risk)과 회복력·복구우선순위(hp_resilience) 산정에 공통으로
필요한 전처리를 수행한다.

  [C-1] 10분 간격 침수해석 결과(ASC) 판독 및 건물 중심점·도로 노드 침수심 추출
  [C-2] 누적 최대침수심 갱신
  [C-3] 건물 자산가치(구조물·내용물) 산정
  [C-4] 침수심 기반 구조물·내용물 취약도 및 피해액 산정 (현재시점 / 최대시점)
  [C-5] 건물 용도가중치, 지하 침수위험, 인구지표(lbl/alllbl) 정규화

ver 2.0(HP_timeseries.ipynb, HP_건물_V3.ipynb)의 연산 로직을 그대로 옮긴 것이며,
계산식과 상수는 변경하지 않았다. ver 3.0 에서는 행 단위 반복(iterrows)이던 건물
전처리를 열 단위 연산으로 바꾸었다(산출값 동일). 원본 반복 구현은 *_loop 함수로 보존한다.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

import hp_config as cfg

warnings.filterwarnings('ignore')


# =============================================================================
# 공통 유틸리티
# =============================================================================
def safe_div(a, b):
    """0 나누기 및 결측 방어 (ver 2.0 HP_건물_V3 와 동일)"""
    try:
        f_a = float(a)
        f_b = float(b)
        if f_b == 0.0 or pd.isna(f_b):
            return 0.0
        return f_a / f_b
    except Exception:
        return 0.0


def load_csv_safe(filepath):
    try:
        return pd.read_csv(filepath, encoding='cp949', low_memory=False)
    except Exception:
        return pd.read_csv(filepath, encoding='utf-8', low_memory=False)


def extract_depth_from_asc(asc_path, x_coords, y_coords):
    """ASC 격자 침수심에서 지정 좌표의 침수심을 추출 (원본과 동일)"""
    with open(asc_path, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])
        data = np.loadtxt(f).reshape((int(header['nrows']), int(header['ncols'])))

    xll = header['xllcorner']
    yll = header['yllcorner']
    cs = header['cellsize']
    nodata = header['nodata_value']
    max_y = yll + header['nrows'] * cs

    cols = ((x_coords - xll) // cs).astype(int)
    rows = ((max_y - y_coords) // cs).astype(int)
    valid = (cols >= 0) & (cols < header['ncols']) & (rows >= 0) & (rows < header['nrows'])
    values = np.zeros(len(x_coords), dtype=float)
    sampled = data[rows[valid], cols[valid]]
    sampled[sampled == nodata] = 0.0
    values[valid] = sampled
    return values


# =============================================================================
# [C-0] 기초자료 적재 (회차 시작 시 1회만 수행)
# =============================================================================
def load_static_data():
    """건물·도로 기초자료와 고정입력자료를 적재한다.

    시간단계마다 반복되지 않는 1회성 작업이므로 회차 시작 시 1회만 수행하고,
    소요시간은 시간단계별 구동시간에 포함하지 않는다.
    """
    S = {}

    # --- 건물 기초자료 ------------------------------------------------------
    df_bd_base = load_csv_safe(cfg.SRC_BUILDING_BASE)
    S['bd_x'] = df_bd_base['Centroid_X_5186'].values
    S['bd_y'] = df_bd_base['Centroid_Y_5186'].values
    df_bd_base.columns = [c.lower() if c.lower() in
                          ['bdtyp', 'bdstr', 'bd_f_area', 'use_year', 'id', 'depth'] else c
                          for c in df_bd_base.columns]
    S['df_bd_base'] = df_bd_base

    # --- 도로 기초자료 ------------------------------------------------------
    df_rd_base = load_csv_safe(cfg.SRC_ROAD_BASE)
    gdf_rd = gpd.GeoDataFrame(df_rd_base,
                              geometry=gpd.points_from_xy(df_rd_base['x'], df_rd_base['y']),
                              crs='EPSG:4326').to_crs('EPSG:5186')
    S['df_rd_base'] = df_rd_base
    S['rd_x'] = gdf_rd.geometry.x.values
    S['rd_y'] = gdf_rd.geometry.y.values

    # --- 건물계열 고정입력자료 ----------------------------------------------
    S['adm_income']         = load_csv_safe(cfg.F_ADM_INCOME)
    S['building_cost']      = load_csv_safe(cfg.F_BUILDING_COST)
    S['house_cost']         = load_csv_safe(cfg.F_HOUSE_COST)
    S['building_depreciation'] = load_csv_safe(cfg.F_BUILDING_DEPREC)
    S['house_depreciation'] = load_csv_safe(cfg.F_HOUSE_DEPREC)
    S['building_csvr']      = load_csv_safe(cfg.F_BUILDING_CSVR)
    S['first_floor_elev']   = load_csv_safe(cfg.F_FIRST_FLOOR_ELEV)
    S['building_str_vul']   = load_csv_safe(cfg.F_BUILDING_STR_VUL)
    S['house_str_vul']      = load_csv_safe(cfg.F_HOUSE_STR_VUL)
    S['cont_vul_slope']     = load_csv_safe(cfg.F_CONT_VUL_SLOPE)
    S['cont_vul_b']         = load_csv_safe(cfg.F_CONT_VUL_B)
    S['building_weight']    = load_csv_safe(cfg.F_BUILDING_WEIGHT)
    S['victims_input']      = pd.read_csv(cfg.F_VICTIMS_INPUT, encoding='utf8')
    S['shelter_adm']        = pd.read_csv(cfg.F_SHELTER_ADM, encoding='utf8')

    # --- 도로계열 고정입력자료 ----------------------------------------------
    S['link_df']       = load_csv_safe(cfg.F_LINK)
    S['all_node_df']   = load_csv_safe(cfg.F_ALL_NODE)
    S['main_df']       = load_csv_safe(cfg.F_MAIN_NODE)
    S['sig_df']        = load_csv_safe(cfg.F_SIG)
    S['no_node']       = load_csv_safe(cfg.F_NODES_REMOVE)
    S['tcc_df']        = load_csv_safe(cfg.F_TCC)

    # --- 누적 최대침수심 배열 초기화 ----------------------------------------
    S['max_depth_bd'] = np.zeros(len(S['bd_x']), dtype=float)

    return S


def reset_accumulator(S):
    """회차 시작 시 누적 최대침수심을 초기화한다."""
    S['max_depth_bd'] = np.zeros(len(S['bd_x']), dtype=float)
    return S


# =============================================================================
# [C-1 ~ C-2] 침수심 추출
# =============================================================================
def extract_step_depth(S, asc_path):
    """10분 단위 ASC 1개에서 건물·도로 침수심을 추출하고 누적 최대침수심을 갱신"""
    depth_bd = extract_depth_from_asc(asc_path, S['bd_x'], S['bd_y'])
    depth_rd = extract_depth_from_asc(asc_path, S['rd_x'], S['rd_y'])

    S['max_depth_bd'] = np.maximum(S['max_depth_bd'], depth_bd)

    df_bd = S['df_bd_base'].copy()
    df_rd = S['df_rd_base'].copy()
    df_bd['depth'] = depth_bd
    df_rd['depth'] = depth_rd

    df_bd_max = df_bd.copy()
    df_bd_max['max_depth'] = S['max_depth_bd']

    return df_bd, df_bd_max, df_rd


# =============================================================================
# [C-3 ~ C-4] 건물 자산가치 · 취약도 · 피해액
# =============================================================================
_COLS_INIT = ['Damage_change', 'persisting_period', 'csvr', 'height', 'depreciation',
              'st_value', 'cont_value', 'f_depth', 'str_vul', 'cont_vul',
              'f_str_damage', 'f_cont_damage', 'total_value', 'f_total_damage']


def _lookup_cost_tables_loop(df, S):
    """건물 용도·구조별 단가, 내용연수, 내용물가치비율, 1층 바닥고 조회 (원본과 동일)"""
    house_cost = S['house_cost']
    house_depreciation = S['house_depreciation']
    building_cost = S['building_cost']
    building_depreciation = S['building_depreciation']
    building_csvr = S['building_csvr']
    first_floor_elev = S['first_floor_elev']

    csvr_row = pd.DataFrame()
    floor_elevation_row = pd.DataFrame()

    for index, row in df.iterrows():
        bdtyp = row['bdtyp']
        if bdtyp == '단독주택':
            bdstr = row['bdstr']
            cost_row = house_cost.loc[house_cost['description'] == bdstr]
            depreciation_row = house_depreciation.loc[house_depreciation['description'] == bdstr]
        else:
            cost_row = building_cost.loc[building_cost['occupancy'] == bdtyp]
            depreciation_row = building_depreciation.loc[building_depreciation['occupancy'] == bdtyp]
            csvr_row = building_csvr.loc[building_csvr['occupancy'] == bdtyp]
            floor_elevation_row = first_floor_elev.loc[first_floor_elev['occupancy'] == bdtyp]

        if not cost_row.empty:
            df.at[index, 'Damage_change'] = float(cost_row.iloc[0]['costs'])
        if not depreciation_row.empty:
            df.at[index, 'persisting_period'] = float(depreciation_row.iloc[0]['persisting_period'])
        if not csvr_row.empty:
            df.at[index, 'csvr'] = float(csvr_row.iloc[0]['csvr'])
        if not floor_elevation_row.empty:
            df.at[index, 'height'] = float(floor_elevation_row.iloc[0]['height'])
    return df


def _asset_value_loop(df, depth_col):
    """감가상각 반영 구조물·내용물 자산가치와 1층 바닥고 보정 침수심 f_depth 산정"""
    for index, row in df.iterrows():
        use_year = pd.to_numeric(row['use_year'], errors='coerce') if pd.notna(row['use_year']) else float(cfg.BASE_YEAR)
        bd_f_area = pd.to_numeric(row['bd_f_area'], errors='coerce') if pd.notna(row['bd_f_area']) else 0.0
        Damage_change = row['Damage_change']
        persisting_period = row['persisting_period']
        csvr = row['csvr']
        height = row['height']

        current_age = cfg.BASE_YEAR - (use_year if pd.notna(use_year) else float(cfg.BASE_YEAR))
        base_val = bd_f_area * Damage_change

        depreciation = safe_div(base_val - (base_val * 0.2), persisting_period)

        if current_age <= persisting_period:
            st_value = base_val * (1 - safe_div((base_val - depreciation * current_age), base_val))
        else:
            st_value = base_val * 0.2

        cont_value = st_value * safe_div(csvr, 100.0)

        f_depth = pd.to_numeric(row[depth_col], errors='coerce') - height
        if pd.isna(f_depth):
            f_depth = 0.0

        df.at[index, 'depreciation'] = depreciation
        df.at[index, 'st_value'] = st_value
        df.at[index, 'cont_value'] = cont_value
        df.at[index, 'total_value'] = cont_value
        df.at[index, 'f_depth'] = f_depth
    return df


_STR_BANDS = ['0m', '0-0.3m', '0.3-0.6m', '0.6-0.9m', '0.9-1.2m', '1.2-1.5m',
              '1.5-1.8m', '1.8-2.1m', '2.1-2.4m', '2.4-2.7m', '2.7-3.0m']


def _band_of(f_depth):
    """침수심 구간(취약도 곡선 구간) 판정 — 원본 if/elif 사슬과 동일"""
    if f_depth == 0:
        return '0m'
    elif 0 < f_depth < 0.3:
        return '0-0.3m'
    elif 0.3 <= f_depth < 0.6:
        return '0.3-0.6m'
    elif 0.6 <= f_depth < 0.9:
        return '0.6-0.9m'
    elif 0.9 <= f_depth < 1.2:
        return '0.9-1.2m'
    elif 1.2 <= f_depth < 1.5:
        return '1.2-1.5m'
    elif 1.5 <= f_depth < 1.8:
        return '1.5-1.8m'
    elif 1.8 <= f_depth < 2.1:
        return '1.8-2.1m'
    elif 2.1 <= f_depth < 2.4:
        return '2.1-2.4m'
    elif 2.4 <= f_depth < 2.7:
        return '2.4-2.7m'
    else:
        return '2.7-3.0m'


def _vulnerability_loop(df, S):
    """구조물 취약도 str_vul, 내용물 취약도 cont_vul 산정 (원본과 동일)"""
    house_str_vul = S['house_str_vul']
    building_str_vul = S['building_str_vul']
    cont_vul_slope = S['cont_vul_slope']
    cont_vul_b = S['cont_vul_b']

    for index, row in df.iterrows():
        bdtyp = row['bdtyp']
        f_depth = row['f_depth']
        if bdtyp == '단독주택':
            bdstr = row['bdstr']
            structure_vul_row = house_str_vul.loc[house_str_vul['description'] == bdstr]
            content_vul_slope_row = cont_vul_slope.loc[cont_vul_slope['occupancy'] == bdtyp].copy()
            content_vul_b_row = cont_vul_b.loc[cont_vul_b['occupancy'] == bdtyp].copy()
        else:
            structure_vul_row = building_str_vul.loc[building_str_vul['occupancy'] == bdtyp]
            content_vul_slope_row = cont_vul_slope.loc[cont_vul_slope['occupancy'] == bdtyp]
            content_vul_b_row = cont_vul_b.loc[cont_vul_b['occupancy'] == bdtyp]

        band = _band_of(f_depth)

        if not structure_vul_row.empty:
            df.at[index, 'str_vul'] = float(structure_vul_row.iloc[0][band])

        if not content_vul_slope_row.empty:
            slope = content_vul_slope_row.iloc[0][band]
            b = content_vul_b_row.iloc[0][band]
            df.at[index, 'cont_vul'] = float(slope * f_depth + b)

    df['vul'] = df['cont_vul']
    return df


def _damage_loop(df):
    """구조물·내용물 피해액 산정 (원본과 동일)"""
    for index, row in df.iterrows():
        str_value = float(row['st_value'])
        str_vul = float(row['str_vul'])
        cont_value = float(row['cont_value'])
        cont_vul = float(row['cont_vul'])
        df.at[index, 'f_str_damage'] = str_value * safe_div(str_vul, 100.0)
        df.at[index, 'f_cont_damage'] = cont_value * safe_div(cont_vul, 100.0)
        df.at[index, 'f_total_damage'] = cont_value * safe_div(cont_vul, 100.0)
    return df


# =============================================================================
# 벡터화 구현 (원본 행 단위 루프와 결과 동일 — 구동시간 단축)
#   원본 루프 구현은 위의 *_loop 함수로 보존하며, 동등성 검증에 사용한다.
# =============================================================================
_BAND_IDX = {b: i for i, b in enumerate(_STR_BANDS)}


def _sdiv(a, b):
    """safe_div 의 벡터화 판. b 가 0 이거나 결측이면 0.0 (스칼라 b 도 허용)"""
    a = pd.to_numeric(pd.Series(a) if not isinstance(a, pd.Series) else a, errors='coerce')
    if not isinstance(b, pd.Series):
        b = pd.Series(np.full(len(a), float(b)), index=a.index)
    b = pd.to_numeric(b, errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b.replace(0.0, np.nan)
    return out.where(b.notna() & (b != 0.0), 0.0)


def _first_map(tbl, key_col, val_col):
    """조회표를 '첫 일치 행' 기준 매핑으로 변환 (원본 .iloc[0] 과 동일)"""
    t = tbl.drop_duplicates(subset=[key_col])
    return pd.Series(pd.to_numeric(t[val_col], errors='coerce').to_numpy(),
                     index=t[key_col].to_numpy())


def _mapped(keys, mapping):
    return pd.to_numeric(pd.Series(keys).map(mapping), errors='coerce').to_numpy()


def _carry_lookup(df, is_house, tbl, key_col, val_col):
    """비(非)단독주택 행에서만 갱신되고 단독주택 행은 직전 값을 이월하는 조회.

    원본 루프에서 csvr_row / floor_elevation_row 가 else 분기에서만 갱신되어,
    단독주택 행은 직전 비단독주택 행의 조회 결과를 그대로 사용하던 동작을 재현한다.
    조회 실패(빈 결과)도 그대로 이월되므로 NaN 상태가 유지된다.
    """
    mapping = _first_map(tbl, key_col, val_col)
    val = pd.Series(np.nan, index=df.index, dtype='float64')
    upd = ~is_house
    val.loc[upd] = _mapped(df.loc[upd, 'bdtyp'].to_numpy(), mapping)
    grp = upd.cumsum()
    return val.groupby(grp).transform('first')


def _lookup_cost_tables(df, S):
    """건물 용도·구조별 단가, 내용연수, 내용물가치비율, 1층 바닥고 조회 (벡터화)"""
    is_house = df['bdtyp'].eq('단독주택')

    cost = pd.Series(np.nan, index=df.index, dtype='float64')
    cost.loc[is_house] = _mapped(df.loc[is_house, 'bdstr'].to_numpy(),
                                 _first_map(S['house_cost'], 'description', 'costs'))
    cost.loc[~is_house] = _mapped(df.loc[~is_house, 'bdtyp'].to_numpy(),
                                  _first_map(S['building_cost'], 'occupancy', 'costs'))
    df['Damage_change'] = cost.fillna(df['Damage_change'])

    pp = pd.Series(np.nan, index=df.index, dtype='float64')
    pp.loc[is_house] = _mapped(df.loc[is_house, 'bdstr'].to_numpy(),
                               _first_map(S['house_depreciation'], 'description', 'persisting_period'))
    pp.loc[~is_house] = _mapped(df.loc[~is_house, 'bdtyp'].to_numpy(),
                                _first_map(S['building_depreciation'], 'occupancy', 'persisting_period'))
    df['persisting_period'] = pp.fillna(df['persisting_period'])

    df['csvr'] = _carry_lookup(df, is_house, S['building_csvr'],
                               'occupancy', 'csvr').fillna(df['csvr'])
    df['height'] = _carry_lookup(df, is_house, S['first_floor_elev'],
                                 'occupancy', 'height').fillna(df['height'])
    return df


def _asset_value(df, depth_col):
    """감가상각 반영 자산가치와 1층 바닥고 보정 침수심 f_depth 산정 (벡터화)"""
    raw_uy = df['use_year']
    use_year = pd.to_numeric(raw_uy, errors='coerce').where(raw_uy.notna(), float(cfg.BASE_YEAR))
    use_year = use_year.fillna(float(cfg.BASE_YEAR))
    raw_area = df['bd_f_area']
    area = pd.to_numeric(raw_area, errors='coerce').where(raw_area.notna(), 0.0)

    dc = pd.to_numeric(df['Damage_change'], errors='coerce')
    pp = pd.to_numeric(df['persisting_period'], errors='coerce')
    csvr = pd.to_numeric(df['csvr'], errors='coerce')
    height = pd.to_numeric(df['height'], errors='coerce')

    current_age = cfg.BASE_YEAR - use_year
    base_val = area * dc
    depreciation = _sdiv(base_val - (base_val * 0.2), pp)

    st_value = (base_val * (1 - _sdiv(base_val - depreciation * current_age, base_val))
                ).where(current_age <= pp, base_val * 0.2)
    cont_value = st_value * _sdiv(csvr, 100.0)

    f_depth = (pd.to_numeric(df[depth_col], errors='coerce') - height).fillna(0.0)

    df['depreciation'] = depreciation
    df['st_value'] = st_value
    df['cont_value'] = cont_value
    df['total_value'] = cont_value
    df['f_depth'] = f_depth
    return df


def _band_series(f_depth):
    """침수심 구간 판정 (원본 if/elif 사슬과 동일, 결측은 마지막 구간)"""
    fd = pd.to_numeric(f_depth, errors='coerce')
    edges = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
    conds = [fd == 0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        conds.append((fd > lo) & (fd < hi) if lo == 0.0 else (fd >= lo) & (fd < hi))
    return pd.Series(np.select(conds, _STR_BANDS[:-1], default=_STR_BANDS[-1]),
                     index=f_depth.index)


def _pick_band(tbl, key_col, keys, band_idx):
    """조회표에서 (키, 침수심 구간) 에 해당하는 값을 벡터로 추출"""
    lut = tbl.drop_duplicates(subset=[key_col]).set_index(key_col).reindex(columns=_STR_BANDS)
    arr = lut.reindex(pd.Index(keys)).to_numpy(dtype='float64')
    return arr[np.arange(len(keys)), band_idx]


def _vulnerability(df, S):
    """구조물 취약도 str_vul, 내용물 취약도 cont_vul 산정 (벡터화)"""
    n = len(df)
    is_house = df['bdtyp'].eq('단독주택').to_numpy()
    band = _band_series(df['f_depth'])
    bidx = band.map(_BAND_IDX).to_numpy()
    fd = pd.to_numeric(df['f_depth'], errors='coerce').to_numpy()

    sv = np.full(n, np.nan)
    if is_house.any():
        sv[is_house] = _pick_band(S['house_str_vul'], 'description',
                                  df['bdstr'].to_numpy()[is_house], bidx[is_house])
    if (~is_house).any():
        sv[~is_house] = _pick_band(S['building_str_vul'], 'occupancy',
                                   df['bdtyp'].to_numpy()[~is_house], bidx[~is_house])
    df['str_vul'] = pd.Series(sv, index=df.index).fillna(df['str_vul'])

    # 내용물 취약도는 단독주택·비단독주택 모두 bdtyp 로 조회 (원본과 동일)
    slope = _pick_band(S['cont_vul_slope'], 'occupancy', df['bdtyp'].to_numpy(), bidx)
    inter = _pick_band(S['cont_vul_b'], 'occupancy', df['bdtyp'].to_numpy(), bidx)
    df['cont_vul'] = pd.Series(slope * fd + inter, index=df.index).fillna(df['cont_vul'])

    df['vul'] = df['cont_vul']
    return df


def _damage(df):
    """구조물·내용물 피해액 산정 (벡터화)"""
    st_value = pd.to_numeric(df['st_value'], errors='coerce')
    cont_value = pd.to_numeric(df['cont_value'], errors='coerce')
    df['f_str_damage'] = st_value * _sdiv(df['str_vul'], 100.0)
    cont_damage = cont_value * _sdiv(df['cont_vul'], 100.0)
    df['f_cont_damage'] = cont_damage
    df['f_total_damage'] = cont_damage
    return df


def preprocess_building(df_bd, df_bd_max, S):
    """건물 공통 전처리 : 자산가치 → 취약도 → 피해액 → 용도가중치 → 지하위험 → 인구지표

    Returns
    -------
    building_cur : 현재 시점(해당 10분 시간단계) 건물 테이블
    building_max : 누적 최대침수심 기준 건물 테이블
    """
    adm_income = S['adm_income']
    building_weight = S['building_weight']

    building_max = df_bd_max.copy()
    building_cur = df_bd.copy()
    building_ep = df_bd.copy()

    building_cur['지하코드'] = building_cur['bdtyp'].apply(
        lambda x: 1 if x in ['단독주택', '다세대주택/연립주택', '아파트'] else 0)

    # 행정동 소득(income) — ver 2.0 과 같은 열 위치('지하코드' 바로 뒤)가 되도록 먼저 부여
    inc = adm_income.set_index('ADM_CD')['income']
    building_max['income'] = building_max['ADM_CD'].map(inc).fillna(0.0)
    building_cur['income'] = building_cur['ADM_CD'].map(inc).fillna(0.0)
    building_ep['income'] = building_ep['ADM_CD'].map(inc).fillna(0.0)

    for col in _COLS_INIT:
        building_max[col] = 0.0
        building_cur[col] = 0.0

    # ---- 최대침수심 기준 --------------------------------------------------
    building_max = _lookup_cost_tables(building_max, S)
    building_max = _asset_value(building_max, depth_col='max_depth')
    building_max.loc[(building_max['f_depth'] == 0) | (building_max['f_depth'].isna()), 'f_depth'] = 0.3
    building_max.loc[building_max['f_depth'] < 0, 'f_depth'] = 0.0
    building_max = _vulnerability(building_max, S)
    building_max = _damage(building_max)

    # ---- 인구지표(lbl / alllbl) 정규화 ------------------------------------
    gid_counts = building_max['gid'].value_counts().to_dict()
    gid_counts_all = building_max['allgid'].value_counts().to_dict()
    building_max['lbl'] = _sdiv(building_max['lbl'],
                                building_max['gid'].map(gid_counts).fillna(1))
    building_max['alllbl'] = _sdiv(building_max['alllbl'],
                                   building_max['allgid'].map(gid_counts_all).fillna(1))

    lbl_map = building_max.set_index('id')['lbl'].to_dict()
    alllbl_map = building_max.set_index('id')['alllbl'].to_dict()
    def _remap(frame, col, mapping):
        has = frame['id'].isin(mapping.keys())
        return frame['id'].map(mapping).where(has, frame[col])

    building_cur['lbl'] = _remap(building_cur, 'lbl', lbl_map)
    building_cur['alllbl'] = _remap(building_cur, 'alllbl', alllbl_map)
    building_ep['lbl'] = _remap(building_ep, 'lbl', lbl_map)
    building_ep['alllbl'] = _remap(building_ep, 'alllbl', alllbl_map)

    building_ep['EP'] = (building_ep['lbl'] * building_ep['income'] * 8.0).fillna(0.0)

    # ---- 현재 시점 기준 ---------------------------------------------------
    building_cur = _lookup_cost_tables(building_cur, S)
    building_cur = _asset_value(building_cur, depth_col='depth')
    building_cur.loc[building_cur['f_depth'] < 0, 'f_depth'] = 0.0
    building_cur = _vulnerability(building_cur, S)

    building_cur.loc[building_cur['f_depth'] == 0, 'T_weight'] = 8.0
    building_cur.loc[building_cur['f_depth'] > 0, 'T_weight'] = 0.0
    building_cur['EP'] = (building_cur['lbl'] * building_cur['income'] * building_cur['T_weight']).fillna(0.0)
    building_cur['PL'] = building_cur['lbl'] * building_cur['income'] * 8.0 - building_cur['EP'].fillna(0.0)

    building_cur = _damage(building_cur)

    # ---- 용도가중치 -------------------------------------------------------
    building_cur['weight_word'] = pd.to_numeric(
        building_cur['bdtyp'].map(_first_map(building_weight, 'occupancy', 'weight_word')),
        errors='coerce')

    # ---- 지하 침수위험 ----------------------------------------------------
    building_cur['under_risk'] = 0.0
    mask = building_cur['지하코드'] == 1
    num_series = (1.59 * 0.9 * (pd.to_numeric(building_cur.loc[mask, 'f_depth'],
                                              errors='coerce') ** 1.65)) * 60.0 * 60.0
    den_series = pd.to_numeric(building_cur.loc[mask, 'bd_f_area'], errors='coerce') * 0.4
    building_cur.loc[mask, 'under_risk'] = num_series / den_series.replace(0.0, np.nan)
    building_cur['under_risk'] = building_cur['under_risk'].fillna(0.0)
    building_cur.loc[building_cur['under_risk'] > 1, 'under_risk'] = 1.0

    return building_cur, building_max


def preprocess_road(df_rd):
    """도로 공통 전처리 : 컬럼명 정규화 및 통행 가능 수심 기준 적용 준비"""
    road = df_rd.copy()
    road.columns = [c.lower() if c.lower() == 'depth' else c for c in road.columns]
    if 'depth' not in road.columns:
        road['depth'] = 0.0
    road['depth'] = pd.to_numeric(road['depth'], errors='coerce').fillna(0.0).clip(lower=0.0)
    return road


# =============================================================================
# 시간단계 1개에 대한 공통 전처리 일괄 수행
# =============================================================================
def run_common(S, asc_path):
    """10분 단위 입력자료 1개에 대한 공통 전처리 전체 수행

    Returns
    -------
    dict : {'building_cur', 'building_max', 'road', 'depth_raw_bd', 'depth_raw_rd'}
    """
    df_bd, df_bd_max, df_rd = extract_step_depth(S, asc_path)
    building_cur, building_max = preprocess_building(df_bd, df_bd_max, S)
    road = preprocess_road(df_rd)
    return {
        'building_cur': building_cur,
        'building_max': building_max,
        'road': road,
    }
