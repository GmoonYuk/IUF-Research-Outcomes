# -*- coding: utf-8 -*-
"""
HP 모형 ver 3.0 — 시계열 구동 러너 (hp_runner)
====================================================================
10분 간격 침수해석 결과(ASC)를 시간 순서대로 읽어
  1단계 공통 전처리 → 2단계 침수위험등급 → 3단계 회복력·복구우선순위
를 수행하고, 단계별 소요시간을 함께 기록한다.

ver 2.0 의 HP_timeseries.ipynb 가 시간단계마다 HP_건물_V3.ipynb · HP_도로.ipynb 를
별도 커널(jupyter nbconvert)로 실행하고 중간 CSV 로 자료를 주고받던 것을,
같은 프로세스 안에서 함수 호출과 메모리 상 DataFrame 전달로 바꾼 것이다.
산출 결과파일의 이름·열 순서는 ver 2.0 과 동일하다.

구동시간 정의
  T_RISK(k) = T_C(k) + T_G(k)   : 입력자료 1개 → 침수위험등급 산출 완료
  T_RES(k)  =          T_R(k)   : 위험등급 산출물 → 회복력·복구우선순위 산출 완료
  T_ALL(k)  = T_RISK + T_RES    : 통합 파이프라인 1스텝
"""
import os
import re
import time
import hashlib

import hp_config as cfg
import hp_common as common
import hp_risk as risk
import hp_resilience as res

# 시간단계별 필수 결과파일 (ver 2.0 과 동일한 6종 + ver 3.0 추가 2종)
RISK_FILES = ["results_building_risk.csv", "results_road_risk.csv", "region_risk_grade.csv"]
RES_FILES = ["results_building.csv", "results_road.csv", "region_results_final.csv",
             "building_risk_final.csv", "Priority.csv"]


def clock():
    """단조시계(monotonic). 시스템 시각 변경의 영향을 받지 않는다."""
    return time.perf_counter()


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def list_asc_files(asc_dir=None, n_steps=None):
    """H_ 로 시작하는 ASC 파일을 파일명 내 숫자 오름차순으로 정렬하여 반환"""
    asc_dir = asc_dir or cfg.ASC_INPUT_DIR
    files = [f for f in os.listdir(asc_dir) if f.startswith('H_') and f.endswith('.asc')]
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    if n_steps is not None:
        files = files[:n_steps]
    return files


def prepare_context(build_road_baseline=True, n_jobs=-1):
    """회차 시작 시 1회만 수행하는 준비작업 — 시간단계별 구동시간에 미포함

    (1) 기초자료·학습모델 적재
    (2) 도로망 기준값 산정 (침수 여부와 무관한 침수 전 최단경로 등)

    두 작업 모두 10분 단위 입력자료가 바뀌어도 결과가 변하지 않으므로
    회차당 1회만 수행하고, 소요시간은 별도로 기록한다.
    """
    t0 = clock()
    S = common.load_static_data()
    risk_models = risk.load_risk_models()
    resilience_models = res.load_resilience_models()
    t1 = clock()

    base_seconds = 0.0
    if build_road_baseline:
        res.get_road_baseline(S, n_jobs=n_jobs)
        base_seconds = round(clock() - t1, 3)

    return {'S': S, 'risk_models': risk_models, 'resilience_models': resilience_models,
            'load_seconds': round(t1 - t0, 3),
            'road_baseline_seconds': base_seconds,
            'setup_seconds': round(clock() - t0, 3)}


def run_step(ctx, asc_name, out_dir=None, save=True, n_jobs=-1, asc_dir=None):
    """10분 단위 입력자료 1개에 대한 HP 모형 구동 및 단계별 계측

    Returns
    -------
    timing : dict  단계별 소요시간과 정상종료 판정
    result : dict  {'common', 'risk', 'resilience'}
    """
    asc_dir = asc_dir or cfg.ASC_INPUT_DIR
    asc_path = os.path.join(asc_dir, asc_name)
    step_name = asc_name.replace('.asc', '')
    step_dir = os.path.join(out_dir, step_name) if (out_dir and save) else None

    S = ctx['S']
    ok_risk, ok_res, err = 1, 1, ""

    # ---------------- 1·2단계 : 공통 전처리 + 침수위험등급 -------------------
    t_s = clock()
    try:
        co = common.run_common(S, asc_path)
    except Exception as e:
        ok_risk = 0
        err += f"[공통전처리] {type(e).__name__}: {e} "
        co = None
    t_c = clock()

    risk_out = None
    if ok_risk:
        try:
            risk_out = risk.run_risk(co, ctx['risk_models'], out_dir=step_dir, step_name=step_name)
        except Exception as e:
            ok_risk = 0
            err += f"[위험등급] {type(e).__name__}: {e} "
    t_g = clock()

    # ---------------- 3단계 : 회복력 · 복구우선순위 --------------------------
    res_out = None
    if ok_risk:
        try:
            res_out = res.run_resilience(co, risk_out, S, ctx['resilience_models'],
                                         out_dir=step_dir, step_name=step_name, n_jobs=n_jobs)
        except Exception as e:
            ok_res = 0
            err += f"[회복력] {type(e).__name__}: {e} "
    else:
        ok_res = 0
    t_r = clock()

    # ---------------- 필수 결과파일 생성 확인 ------------------------------
    files_risk = files_res = 1
    if step_dir:
        produced = os.listdir(step_dir) if os.path.isdir(step_dir) else []
        files_risk = int(all(any(f.endswith(s) for f in produced) for s in RISK_FILES))
        files_res = int(all(any(f.endswith(s) for f in produced) for s in RES_FILES))

    res_t = (res_out or {}).get('timing', {}) if isinstance(res_out, dict) else {}
    timing = {
        "입력자료": step_name,
        "T_C_공통전처리": round(t_c - t_s, 3),
        "T_G_위험등급": round(t_g - t_c, 3),
        "T_R_회복력": round(t_r - t_g, 3),
        # 3단계 내부 구간
        "T_R1_입력준비": res_t.get('T_P'),
        "T_R2_건물회복력": res_t.get('T_B'),
        "T_R3_도로망분석": res_t.get('T_R'),
        "T_R4_병합저장": res_t.get('T_M'),
        "T_RISK_위험등급_구동시간": round(t_g - t_s, 3),
        "T_RES_회복력_구동시간": round(t_r - t_g, 3),
        "T_ALL_통합": round(t_r - t_s, 3),
        "위험등급_정상종료": int(ok_risk and files_risk),
        "회복력_정상종료": int(ok_res and files_res),
        "비고": err.strip(),
    }
    result = {'common': co, 'risk': risk_out, 'resilience': res_out}
    return timing, result


def run_timeseries(asc_dir=None, out_dir=None, n_steps=None, n_jobs=-1, verbose=True):
    """ASC 폴더 전체를 시간 순서대로 구동하고 시간단계별 로그(DataFrame)를 반환한다.

    노트북(HP_timeseries.ipynb) 없이 스크립트로 구동할 때 사용한다.
      python hp_runner.py                      # hp_config 의 ASC_INPUT_DIR 전체
      python hp_runner.py <ASC폴더> <결과폴더>  # 폴더 지정
    """
    import pandas as pd
    asc_dir = asc_dir or cfg.ASC_INPUT_DIR
    out_dir = out_dir or cfg.ensure_output_dir("TimeSeries")
    os.makedirs(out_dir, exist_ok=True)

    asc_files = list_asc_files(asc_dir=asc_dir, n_steps=n_steps)
    if not asc_files:
        raise FileNotFoundError(f"'H_' 로 시작하는 .asc 파일이 없습니다: {asc_dir}")
    if verbose:
        print(f"입력 ASC {len(asc_files)}개  ({asc_files[0]} ~ {asc_files[-1]})")
        print("결과 저장 :", out_dir)

    ctx = prepare_context(n_jobs=n_jobs)
    if verbose:
        print(f"기초자료·학습모델 적재 : {ctx['load_seconds']:.3f} 초")
        print(f"도로망 기준값 산정     : {ctx['road_baseline_seconds']:.3f} 초")

    logs = []
    for k, name in enumerate(asc_files, 1):
        timing, _ = run_step(ctx, name, out_dir=out_dir, save=True, n_jobs=n_jobs, asc_dir=asc_dir)
        timing["시간단계_k"] = k
        logs.append(timing)
        if verbose:
            flag = "OK " if (timing["위험등급_정상종료"] and timing["회복력_정상종료"]) else "FAIL"
            print(f"[{flag}] k={k:>3}/{len(asc_files)}  {timing['입력자료']:<12} "
                  f"위험등급={timing['T_RISK_위험등급_구동시간']:7.2f}s  "
                  f"회복력={timing['T_RES_회복력_구동시간']:7.2f}s  "
                  f"합계={timing['T_ALL_통합']:7.2f}s")

    df_log = pd.DataFrame(logs)
    df_log["회차준비_기초자료적재_s"] = ctx['load_seconds']
    df_log["회차준비_도로망기준값_s"] = ctx['road_baseline_seconds']
    log_path = os.path.join(out_dir, "HP_시계열구동_로그.csv")
    df_log.to_csv(log_path, index=False, encoding='utf-8-sig')
    if verbose:
        print("\n[저장]", log_path)
    return df_log


if __name__ == "__main__":
    import sys
    a_dir = sys.argv[1] if len(sys.argv) > 1 else None
    o_dir = sys.argv[2] if len(sys.argv) > 2 else None
    run_timeseries(asc_dir=a_dir, out_dir=o_dir)
