# 학습 기록 (Colab)

Colab GPU에서 Ultralytics YOLO26n을 학습했습니다. Colab 노트북 파일은 보관하지 않았으므로, 아래 레시피는 학습 산출물 [`training/args.yaml`](training/args.yaml)에 남은 값 그대로입니다.

## 이 저장소의 `training/`은 v1 학습 결과입니다

| 항목 | 값 (args.yaml) |
|---|---|
| model | `yolo26n.pt` (COCO pretrained에서 시작) |
| data | `/content/amet_track/data.yaml` (v1 export) |
| epochs / patience | 60 / 20 |
| batch / imgsz | 16 / 640 |
| optimizer | auto |
| seed / deterministic | 0 / true |
| close_mosaic / amp | 10 / true |

마지막 epoch의 validation(v1 valid 35장): **P 0.785 / R 0.683 / mAP50 0.673 / mAP50-95 0.477** ([`training/results.csv`](training/results.csv), 곡선·혼동행렬은 [`training/`](training)).

```bash
# Colab (재현용, 대략)
pip install ultralytics
yolo detect train model=yolo26n.pt data=/content/amet_track/data.yaml \
  epochs=60 patience=20 imgsz=640 batch=16 seed=0 deterministic=True close_mosaic=10 amp=True
```

## v2 모델 (`old_v2`, 실차에 올린 모델)

- v1의 `best.pt`에서 이어 v2 데이터셋(원본 546장)으로 다시 학습했습니다(학습 기록 기준).
- 54장 held-out test 결과: `old_v2` **P 0.836 / R 0.787 / mAP50 0.846 / mAP50-95 0.661**, `clean_v2` P 0.804 / R 0.776 / mAP50 0.813 / mAP50-95 0.639 ([`docs/feature_branch_report_20260826.md`](docs/feature_branch_report_20260826.md)).
- 배포한 ONNX의 SHA-256은 `64706415d3f27f70…`입니다. v2 학습 곡선과 가중치(`.pt`/`.onnx`)는 이 저장소에 싣지 않았습니다.

v1과 v2 수치는 서로 다른 데이터와 평가 세트에서 나온 값이라 직접 비교하지 않습니다.
