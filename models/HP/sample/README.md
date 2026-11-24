# HP 예제 출력

2022년 8월 8일 호우(10분 간격 6시간, 36개 시간단계)를 HP ver 3.0 으로 구동한 결과 가운데 **행정동 단위 파일**만 모은 것입니다.
건물 단위·도로 노드 단위 결과와 입력자료는 넣지 않았습니다([상위 README](../README.md)의 입력 항목 참고).

| 경로 | 내용 |
|---|---|
| `H_130MIN/` | 시간단계 1개(130분, 평균 Resilience 가 가장 낮은 시점)의 행정동 단위 결과 4종 |
| `region_results_final/` | 36개 시간단계의 `H_<분>MIN_region_results_final.csv`. 결과 표출 코드(`../src/analysis/`)의 입력으로 쓸 수 있습니다 |
| `HP_시계열구동_로그.csv` | 36개 시간단계를 1회 구동한 로그 (시간단계별·단계별 소요시간, 시간단계당 평균 약 58초) |
| `analysis1_example.png` ~ `analysis3_example.png` | 결과 표출 코드 출력 예시 (ver 2.0 의 12시간 모의 결과로 그린 그림) |

## H_130MIN/ 의 파일

| 파일 | 행 수 | 내용 |
|---|---|---|
| `H_130MIN_results_building.csv` | 45 | 행정동별 피해액(천원), 재난복구지원금(천원), 예측 이재민, 견고성(건물), 자원부존성, 건물 위험등급 분포 |
| `H_130MIN_results_road.csv` | 49 | 행정동별 도로 노드 위험등급 분포, 침수 전·후 최단경로 수·길이·시간, TCC, RA, SR, RI |
| `H_130MIN_region_risk_grade.csv` | 49 | 행정동별 건물·도로 위험등급 분포 |
| `H_130MIN_region_results_final.csv` | 49 | 위 건물·도로 결과를 행정동 코드로 합치고 Robustness, Rapidness, Redundancy, Resourcefulness, Resilience 를 붙인 것 |

실제 구동에서는 같은 폴더에 건물 단위 3개(`results_building_risk`, `building_risk_final`, `Priority`)와
도로 노드 단위 1개(`results_road_risk`) 파일이 함께 만들어집니다.

## 참고

- CSV 인코딩은 모형 출력 그대로 **cp949** 입니다(로그만 utf-8-sig). GitHub 웹 화면에서는 한글이 깨져 보일 수 있으니 내려받아 엑셀 등으로 여세요.
- 도로 결과에는 건물 자료가 없는 경계 행정동 4곳(신도림동, 가리봉동, 가산동, 당산1동)이 포함되어 49행이며, 이 행정동의 건물 열은 비어 있습니다.
- 결과 표출 노트북은 현재 폴더의 `H_*MIN_region_results_final.csv` 를 읽습니다. `region_results_final/` 의 CSV 를 노트북과 같은 폴더에 복사해 실행하면 됩니다.
