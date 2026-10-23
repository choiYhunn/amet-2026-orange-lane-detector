# Real-camera orange YOLO hybrid report

> **Context (added 2026-10-23).** This is the working report from my personal feature branch
> (`feature/orange-yolo-hybrid-realcar`), written on 2026-08-26 while the candidate was still being
> evaluated. "Official launcher" and "final candidate" below refer to that branch only. The candidate
> was **never merged into the team's main branch and was not used in the race**; the reasons are in the
> [repository README](../README.md). The "318 ms p50 / 400 ms p95" bench figure below is run `fb45865`;
> the later runs `74b3bf1` and `e37cdca` measured about 373–376 ms p50
> ([`realcar_runs/`](../realcar_runs)). Text below is unchanged from the branch.

## 20260826 e37cdca mounted-run correction

The mounted-camera run left the track because the simulator-derived BEV source
quad ignored the real near-field dashes. The first centered frame produced a
false `-0.714 m` lateral node and saturated right steering. Since the orange
divider itself is a legal path, the final candidate now follows ordered YOLO
box centers directly in raw pixels with zero lane offset. BEV steering, cone
lane switching, and the start-checker steering override are disabled in this
mode. The recorded mounted start frame yields `x=222.9/480` and `+1.57 deg`
initial steering instead of `-20 deg`.

## Decision

The candidate is based on commit `b965990` (`CYH R3 LANE-SAFE FAST`), not R6.
R3 completed all three baseline simulator repeats with penalty 0. The tested R6
source baseline completed only one of three repeats, so it was rejected as the
stability base for this real-car perception experiment.

The official real-car launcher enables YOLO as the primary far-road geometry
source. YOLO OFF is retained only for controller regression and emergency
fallback. YOLO quality is evaluated on recorded/mounted real-camera frames; a
simulator YOLO-ON run is not a physical acceptance result.

## Model and runtime

- Model: YOLO26n detect, one `center` class, 640 px, 60 epochs
- Selected model: `old_v2`, deployed from `orange_center_FINAL.onnx`
- ONNX SHA256: `64706415d3f27f70c7e1baafc9efbdabd7034712c63c143f41753653819c399d`
- Default confidence: 0.20
- Contract: input `1x3x640x640`, output `1x300x6`
- Output rows: end-to-end NMS `x1,y1,x2,y2,confidence,class_id`
- Runtime: ONNX Runtime CPU first, OpenCV DNN fallback
- Compatibility: evaluator Docker ONNX Runtime 1.29.0 loads the graph; its
  OpenCV 4.11 build does not support the model's `TopK(K=300)` node
- Windows OpenCV 5 full-recording latency, 596 runs: 53.14 ms mean, 52.09 ms
  p50, 62.46 ms p95 total (47.40 ms mean inference)
- Evaluator Docker ONNX Runtime 1.29.0 under active simulator load, 30 warm
  runs: 112.13 ms mean, 104.81 ms p50, 173.18 ms p95 total (102.64 ms mean
  inference)
- Physical PhysiCar CPU bench result: 318 ms p50 and 400 ms p95 inference in the
  first archived run. This drove the low-speed envelope and the separation of
  YOLO far geometry from current-frame near CTE.

The Windows environment did not have Ultralytics or ONNX Runtime installed, so
a `best.pt` versus ONNX library comparison was not manufactured by adding new
packages. The exported ONNX tensor contract and detections were instead checked
directly with OpenCV 5, then repeated with the evaluator's installed ONNX
Runtime 1.29.0.

The runtime never installs packages or accesses the network at launch. A load
or inference failure leaves the stable tracker in control.

## Measured real-camera results

The first mounted-track archive (`74b3bf1`) exposed a profile mismatch rather
than a motor failure: 332/409 frames were `orange_blind_stop`, because the lit
black mat exceeded the simulator `ROAD_V_MAX=125` and the real orange dashes
fell below simulator `S_MIN=145`. The measured real profile now uses road
`V<=200` and orange `H=8..27, S>=90, V>=100`; three committed mounted fixtures
are 3/3 stable-valid and at least 2/3 YOLO-geometry-valid. Simulator thresholds
remain unchanged.

The complete `rec.zip` recording contained 596 native 480x360 frames.

| Metric | Result |
|---|---:|
| Frames with no detection | 132 / 596 (22.1%) |
| Detections per frame | 5.88 |
| YOLO-derived valid geometry | 289 / 596 (48.5%) |
| Existing stable valid geometry | 205 / 596 (34.4%) |
| Conservative stable-invalid rescues | 101 frames |
| Weak-stable, agreeing YOLO candidates | 62 frames |
| Expected fused valid geometry | 306 / 596 (51.3%) |
| Measured path conflicts | 0 frames |

The committed contact sheet includes straight, curve, cone, blurred, and
off-road samples. It is a diagnostic artifact, not proof of physical driving
performance.

On the same 54-image clean test, the final `old_v2` measured precision 0.836318,
recall 0.787164, mAP50 0.846409, and mAP50-95 0.661311. The rejected `clean_v2`
measured 0.803775, 0.775910, 0.812509, and 0.639240 respectively. At confidence
0.20, `old_v2` averaged 10.56 detections/image with one zero-detection image;
`clean_v2` averaged 8.89 with zero zero-detection images. Because the hybrid
already falls back to stable perception on zero detections, `old_v2` was chosen
for stronger overall precision, recall, and mAP. These A/B values were supplied
from the completed retraining evaluation; the repository validators independently
recheck the selected ONNX artifact, tensor contract, and runtime behavior.

## Fusion policy

The stable HSV/road tracker runs on every controller frame for fresh near-field
heading/CTE and safety fallback. YOLO runs in a single-slot latest-frame worker
every six frames and is accepted as far geometry for at most 0.70 seconds. Its
boxes are treated as measured regions: box centers and orange
components inside each region are transformed through the existing BEV,
road/corridor gates, graph chaining, local-line, and bend estimator.

Fresh, strong YOLO is the far-road authority even when stable is strong. Two
distinct inference results must agree on turn direction before YOLO bend and
latency-compensated turn distance arm the turn state and preview feed-forward.
The first unconfirmed bend candidate already caps speed. Current stable
heading/CTE remains the near-control source between YOLO results; an aged YOLO
path is never treated as current CTE. Stale, empty, weak, missing-model, and
runtime-error results fall back to stable. YOLO therefore leads early 45/90
degree corner perception without converting an old bbox directly into a
steering angle.

## Real start gate

The real profile searches a broad right-side region for a phone-screen circle;
it is not fitted to post-start diagnostic coordinates. A valid green blob with
dark screen surround and circle-like area/aspect wins over unrelated red skin
candidates, then still needs the existing consecutive-frame latch. Simulator
thresholds and decision behavior are unchanged. `source ./run.sh` selects the
real/YOLO defaults on this feature branch, so the official launch remains one
command. Pre-latch raw frames and scores are stored inside each final run ZIP
under `start_gate/`.

The official timeout default is zero (disabled). A 10-second fallback exists
only as the explicit `START_GATE_TIMEOUT_S=10` wheels-clear development option.

## Baseline simulator evidence

R3 direct-python YOLO-OFF baseline:

| Run | Score | Penalty | Outcome |
|---:|---:|---:|---|
| 1 | 39.335 s | 0 | finished |
| 2 | 39.180 s | 0 | finished |
| 3 | 131.395 s | 0 | finished |

Clean completion was 3/3. The large third-run spread is recorded as measured;
it is not hidden by reporting only the best time.

Final evaluator-equivalent `source ./run.sh`, YOLO-OFF acceptance:

| Run | Score | Penalty | Outcome |
|---:|---:|---:|---|
| 1 | 37.675 s | 0 | finished |
| 2 | 37.595 s | 0 | finished |
| 3 | 38.335 s | 0 | finished |

Clean completion was 3/3, with best 37.595 s, average 37.868 s, and worst
38.335 s. The workflow also passed camera fetch, BGR/RGB sanity, orange/road
gates, RED/GO regression, fail-closed start gate, and direct/source entrypoint
equivalence before the three runs.

Post-model-swap evaluator-equivalent `source ./run.sh`, explicitly YOLO OFF:

| Run | Score | Penalty | Outcome |
|---:|---:|---:|---|
| 1 | 39.570 s | 0 | finished |
| 2 | 62.100 s | 1 off-track | finished |
| 3 | 46.695 s | 0 | finished |

This repeat was 2/3 clean. The penalty occurred at simulator time 35.845 s.
The judge command explicitly exported `ORANGE_YOLO_ENABLE=0`, and the exact
stable-observation identity contract passed before the runs; the model was not
loaded and could not affect steering. No path/controller tuning was made in
response. This variance is retained in the report rather than presenting the
model swap as a 3/3-clean driving acceptance.
