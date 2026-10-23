# 라벨링과 데이터셋 버전

Roboflow 워크스페이스 `yhch`, 프로젝트 `amet_track`에서 실차 트랙 이미지를 직접 라벨링했습니다. 원본 이미지와 라벨 파일은 이 저장소에 싣지 않습니다.

## 라벨 규약

- 단일 클래스 **`center`** ([`dataset/data.yaml`](dataset/data.yaml)): 트랙의 **주황색 점선 중앙선 조각(dash) 하나마다 박스 하나**.
- 목적은 차선 전체가 아니라 중앙선 조각의 위치를 검출해 먼 구간의 경로 기하를 얻는 것이었습니다.

## 데이터셋 버전

| 버전 | 원본 이미지 | Roboflow export | 분할 (train / valid / test) | 근거 |
|---|---:|---:|---|---|
| v1 (밝은 조건) | 약 231장 | 575장 (증강 포함) | 517 / 35 / 23 | export README는 [`dataset/`](dataset)에 포함, 원본 수·분할은 학습 기록 |
| v2 (밝음+어두움) | 546장 | 1,364장 (증강 포함, Roboflow v3 export) | 1,228 / 82 / 54 | 학습 기록 (v3 export 파일은 보관하지 않음) |

- export 수에는 학습용 이미지의 증강본(원본 1장당 약 3장)이 포함됩니다. 고유 사진 수가 아닙니다. valid/test는 증강하지 않은 원본입니다.
- v1 export의 전처리·증강: EXIF auto-orientation, 640×640 stretch, 밝기 ±10%, 노출 ±10%, Gaussian blur 0–2.5 px, salt & pepper 0.1% ([`dataset/README.roboflow.txt`](dataset/README.roboflow.txt)).
- v3 export의 라벨 파일에는 bbox 행과 polygon 행이 섞여 있었습니다. 그래서 학습 전에 형식을 점검했고, 원래 라벨로 학습한 `old_v2`와 bbox만 남긴 `clean_v2`를 같은 54장 test로 비교했습니다([`docs/feature_branch_report_20260826.md`](docs/feature_branch_report_20260826.md)).
