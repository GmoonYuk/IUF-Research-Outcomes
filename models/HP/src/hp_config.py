# -*- coding: utf-8 -*-
"""
HP 모형 ver 3.0 — 공통 설정
====================================================================
HP(Hyper Prioritization) 모형 : 10분 간격 침수해석 결과(ASC)로부터
  1단계  공통 전처리         (hp_common.py)     침수심 추출, 자산가치·취약도·피해액
  2단계  침수위험등급 산정    (hp_risk.py)       위험인자 → 위험지수 → 4단계 위험등급
  3단계  회복력·복구우선순위  (hp_resilience.py) 견고성·자원부존성·도로망 분석 → Resilience → 복구우선순위
를 순차 수행한다.

경로는 본 파일 위치 기준 상대경로이므로 폴더째 복사하면 수정이 필요 없다.
다른 위치의 자료를 쓰려면 1절의 네 변수만 절대경로로 바꾸면 된다.
4절의 모형 파라미터는 변경하지 않는다.
"""
import os

# =============================================================================
# 1. 입력·출력 경로
#    hp_config.py 가 있는 폴더 기준 상대경로. 드라이브 문자와 무관하게 동작한다.
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

# 10분 간격 침수해석 결과(ASC) 폴더  ─ H_10MIN.asc ~ H_360MIN.asc
#   · 실시간 운영   : 침수해석 모형이 ASC 를 내려놓는 폴더로 변경
#   · HD 모의 시나리오 : HD 모의결과 ASC 폴더로 변경
ASC_INPUT_DIR = os.path.join(_HERE, "ASC_INPUT", "Sillim_Results")

# 고정입력자료 폴더
DEFAULT_BD = os.path.join(_HERE, "DEFAULT_BD", "defaultdata")     # 건물계열 (21개)
DEFAULT_RD = os.path.join(_HERE, "DEFAULT_RD", "defaultdata2")    # 도로계열 (9개)

# 결과 저장 최상위 폴더  (시간단계별 하위폴더 OUTPUT\TimeSeries\H_(t)MIN\ 생성)
OUTPUT_ROOT = os.path.join(_HERE, "OUTPUT")


def _bd(name):
    return os.path.join(DEFAULT_BD, name)


def _rd(name):
    return os.path.join(DEFAULT_RD, name)


# =============================================================================
# 2. 고정입력자료 파일 (건물계열)
# =============================================================================
SRC_BUILDING_BASE = _bd(r"sinrim building_depth_renew_updated.csv")   # 건물 기초자료(중심점 좌표 포함)

F_ADM_INCOME        = _bd("adm_income(20240212).csv")
F_BUILDING_COST     = _bd("building_cost.csv")
F_HOUSE_COST        = _bd("house_cost.csv")
F_BUILDING_DEPREC   = _bd("building_depreciation.csv")
F_HOUSE_DEPREC      = _bd("house_depreciation.csv")
F_BUILDING_CSVR     = _bd("building_csvr.csv")
F_FIRST_FLOOR_ELEV  = _bd("building_first_floor_elevation0.csv")
F_BUILDING_STR_VUL  = _bd("building_structure_vul.csv")
F_HOUSE_STR_VUL     = _bd("house_structure_vul.csv")
F_CONT_VUL_SLOPE    = _bd("building_content_vul_slope.csv")
F_CONT_VUL_B        = _bd("building_content_vul_b.csv")
F_BUILDING_WEIGHT   = _bd("building_weight.csv")
F_BUILDING_03_DEPTH = _bd("building_03_depth.csv")
F_VICTIMS_INPUT     = _bd("inputdata_2022_inputdata.csv")
F_SHELTER_ADM       = _bd("서울시_이재민임시주거시설_amd.csv")

# 학습모델(.pkl) — 건물계열
P_BD_SCALER   = _bd("scaler.pkl")               # 위험등급 : Hazard/Vulnerability/Exposure 표준화
P_BD_KMEANS   = _bd("kmeans_model.pkl")         # 위험등급 : 위험지수 군집(4단계 등급)
P_RF_VICTIMS  = _bd("disaster_victims_prediction_model_RF.pkl")   # 회복력 : 예측이재민 RF
P_REG_SCALER  = _bd("regression_scaler.pkl")                      # 회복력 : 예측이재민 회귀 표준화
P_REG_EQUATION= _bd("regression_equation_df.pkl")                 # 회복력 : 예측이재민 회귀식

# =============================================================================
# 3. 고정입력자료 파일 (도로계열)
# =============================================================================
SRC_ROAD_BASE = _rd("도림천유역_Boundary_updated.csv")    # 경계 노드 기초자료(x, y 포함)

F_LINK          = _rd("00.expanded_seoul_edges.csv")
F_ALL_NODE      = _rd("00.seoul_nodes_with_centrality.csv")
F_MAIN_NODE     = _rd("00.seoul_main_nodes_adm(20240507).csv")
F_SIG           = _rd("00.SIG.csv")
F_NODES_REMOVE  = _rd("nodes_to_remove.csv")
F_TCC           = _rd("도로별교통혼잡비용_0.csv")

# 학습모델(.pkl) — 도로계열
P_RD_SCALER = _rd("scaler.pkl")          # 위험등급 : 도로 노드 위험지수 표준화
P_RD_KMEANS = _rd("kmeans_model.pkl")    # 위험등급 : 도로 노드 위험등급 군집

# =============================================================================
# 4. 모형 파라미터 (ver 2.0 과 동일 — 변경하지 않음)
# =============================================================================
# --- 공통 --------------------------------------------------------------------
D_PASSABLE = 0.15          # [m] 통행 가능 / 유효 침수 판정 기준 수심
BASE_YEAR  = 2023          # 건물 경과연수 산정 기준연도

# --- 침수위험등급 -------------------------------------------------------------
N_GRADE = 4                # 위험등급 단계 수
GRADE_NAME = {1: "매우낮음", 2: "낮음", 3: "위험", 4: "매우위험"}

# --- 회복력 (도로망 분석) -----------------------------------------------------
TRAVEL_TIME_LIMIT = 0.172      # [hr] 주요 노드 접근성(RA) 판정 기준 통행시간
RECOVERY_RESILIENCE = 0.80     # 기능회복 단계 판정 Resilience 임계값 (결과 해석용)
RECOVERY_DRAINAGE   = 0.85     # 기능회복 단계 판정 배수 축소율 임계값 (결과 해석용)

# --- 예측이재민 판정 임계값 ---------------------------------------------------
VICTIM_RF_TH        = 1349
VICTIM_BLDG_TH      = 161
VICTIM_HOUSE_TH     = 125
VICTIM_UNDER_TH     = 112
VICTIM_POLY_MAX     = 2500
RECOVERY_FUND_PER_BD = 3000000.0   # [원] 건물당 재난복구지원금


def ensure_output_dir(sub=""):
    p = os.path.join(OUTPUT_ROOT, sub) if sub else OUTPUT_ROOT
    os.makedirs(p, exist_ok=True)
    return p
