"""Summarize YOLO inference latency and loop overruns for the archived real-car runs.

Each control frame in telemetry.csv repeats the most recent YOLO result, so latency is
counted once per YOLO result (deduplicated by `yolo_result_frame`). Percentiles use
linear interpolation between order statistics. Standard library only.

Usage: python summarize_runs.py            # prints the table, writes run_summary.csv
"""
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def percentile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(run_dir):
    meta = json.loads((run_dir / 'run_meta.json').read_text(encoding='utf-8'))
    per_result = {}
    with open(run_dir / 'telemetry.csv', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            result_id, ms = row.get('yolo_result_frame'), row.get('yolo_inference_ms')
            if result_id in (None, '', '-1') or ms in (None, ''):
                continue
            if float(ms) > 0:
                per_result[result_id] = float(ms)
    latency = sorted(per_result.values())
    return {
        'run': run_dir.name,
        'control_frames': meta['frames'],
        'loop_overruns': meta['loop_overruns'],
        'completion': meta['completion_reason'],
        'duration_s': meta['duration_s'],
        'yolo_results': len(latency),
        'inference_p50_ms': round(percentile(latency, 0.50), 1),
        'inference_p90_ms': round(percentile(latency, 0.90), 1),
        'inference_p95_ms': round(percentile(latency, 0.95), 1),
        'inference_max_ms': round(latency[-1], 1),
    }


def main():
    rows = [summarize(d) for d in sorted(HERE.iterdir()) if (d / 'telemetry.csv').is_file()]
    with open(HERE / 'run_summary.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['run']}: {r['yolo_results']} results, p50 {r['inference_p50_ms']} ms, "
              f"p95 {r['inference_p95_ms']} ms, overruns {r['loop_overruns']}/{r['control_frames']}, "
              f"ended by {r['completion']}")


if __name__ == '__main__':
    main()
