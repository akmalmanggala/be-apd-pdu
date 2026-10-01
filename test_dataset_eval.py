"""Automated Evaluation Script on PMLD PDU Migas Test Dataset.

Runs comprehensive inference across the 38 test set frames, evaluates detection metrics
(Precision, Recall, F1-Score per APD class), two-stage zoom micro-detection,
and PRD Section 10.6 worker compliance evaluation.
"""

import glob
import json
import os
import time
from collections import defaultdict
import cv2
import numpy as np

from app.config import settings
from app.services.apd_detector import APDDetectorEngine, box_iou, calculate_overlap_ratio

def render_ground_truth(img_bgr: np.ndarray, gt_boxes_by_class: dict) -> np.ndarray:
    """Render ground truth annotations with distinct high-contrast boxes and labels."""
    vis = img_bgr.copy()
    h, w, _ = vis.shape
    apd_colors = {
        "helm": (0, 230, 0),        # Green
        "glove": (0, 140, 255),      # Orange
        "kacamata": (255, 0, 180),   # Magenta
        "sepatu": (255, 220, 0),     # Cyan/Yellow
    }
    for cname, boxes in gt_boxes_by_class.items():
        color = apd_colors.get(cname, (200, 200, 200))
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = f"GT: {cname.upper()}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, max(lh, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Top Banner
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.putText(vis, "GROUND TRUTH (DATASET TEST LABELS)", (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def evaluate_dataset(
    dataset_test_dir: str = r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test",
    output_dir: str = str(settings.OUTPUTS_DIR),
):
    print("=================================================================")
    print("      EVALUASI KOMPREHENSIF TEST SET DATASET APD PDU MIGAS       ")
    print("=================================================================")

    img_dir = os.path.join(dataset_test_dir, "images")
    lbl_dir = os.path.join(dataset_test_dir, "labels")

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Total Citra Test Set: {len(image_paths)} Frame")

    detector = APDDetectorEngine.get_instance()
    class_names = detector.class_names  # ['glove', 'helm', 'kacamata', 'sepatu']

    # Ground truth and detection counters
    gt_counts = {c: 0 for c in class_names}
    tp_counts = {c: 0 for c in class_names}
    fp_counts = {c: 0 for c in class_names}
    fn_counts = {c: 0 for c in class_names}

    # Workers & compliance tracking
    total_workers_screened = 0
    total_compliant_workers = 0
    total_non_compliant_workers = 0
    helm_violations_total = 0
    glove_violations_total = 0
    sepatu_violations_total = 0
    kacamata_violations_total = 0
    total_latency_ms = []

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Read Ground Truth labels
        lbl_file = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
        gt_boxes_by_class = defaultdict(list)
        if os.path.exists(lbl_file):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        cid = int(parts[0])
                        cname = class_names[cid]
                        xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                        x1 = (xc - bw / 2.0) * w
                        y1 = (yc - bh / 2.0) * h
                        x2 = (xc + bw / 2.0) * w
                        y2 = (yc + bh / 2.0) * h
                        gt_boxes_by_class[cname].append([x1, y1, x2, y2])
                        gt_counts[cname] += 1

        # Run pipeline inference (Focus strictly on APD detection without worker boxes)
        t0 = time.perf_counter()
        workers, stats, annotated = detector.detect_image(
            img_bgr=img,
            overlap_threshold=0.80,
            enable_head_zoom=True,
            show_workers=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms.append(elapsed_ms)

        total_workers_screened += stats["total_workers"]
        total_compliant_workers += stats["compliant_workers_count"]
        total_non_compliant_workers += stats["non_compliant_workers_count"]
        helm_violations_total += stats["helm_violations"]
        glove_violations_total += stats["glove_violations"]
        sepatu_violations_total += stats["sepatu_violations"]
        kacamata_violations_total += stats["kacamata_violations"]

        # Collect all detected APDs from the pipeline (global + two-stage head zoom)
        pred_boxes_by_class = defaultdict(list)
        for apd_item in stats.get("all_detected_apds", []):
            cname = apd_item.class_name
            pred_boxes_by_class[cname].append([
                apd_item.bbox.x1,
                apd_item.bbox.y1,
                apd_item.bbox.x2,
                apd_item.bbox.y2,
            ])

        # Evaluate detection matching per class using IoU >= 0.25 (standard for APD objects)
        for cname in class_names:
            gts = gt_boxes_by_class[cname]
            preds = pred_boxes_by_class[cname]

            matched_gts = set()
            for p_box in preds:
                best_iou = 0.0
                best_gi = -1
                for gi, g_box in enumerate(gts):
                    if gi not in matched_gts:
                        iou_val = box_iou(p_box, g_box)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gi = gi

                if best_iou >= 0.25 and best_gi >= 0:
                    matched_gts.add(best_gi)
                    tp_counts[cname] += 1
                else:
                    fp_counts[cname] += 1

            fn_counts[cname] += len(gts) - len(matched_gts)

        # Save annotated images and side-by-side comparisons for ALL 38 frames
        all_frames_dir = os.path.join(output_dir, "all_test_frames")
        stage7_frames_dir = os.path.join(output_dir, "stage7_all_test_frames")
        os.makedirs(all_frames_dir, exist_ok=True)
        os.makedirs(stage7_frames_dir, exist_ok=True)

        eval_path = os.path.join(all_frames_dir, f"eval_frame_{idx:02d}_{filename}")
        cv2.imwrite(eval_path, annotated)
        cv2.imwrite(os.path.join(stage7_frames_dir, f"eval_frame_{idx:02d}_{filename}"), annotated)

        # Generate side-by-side comparison (Ground Truth vs Prediction)
        gt_vis = render_ground_truth(img, gt_boxes_by_class)
        gt_resized = cv2.resize(gt_vis, (1280, 720))
        pred_resized = cv2.resize(annotated, (1280, 720))
        comparison_img = np.hstack([gt_resized, pred_resized])
        comp_path = os.path.join(all_frames_dir, f"comparison_frame_{idx:02d}_{filename}")
        cv2.imwrite(comp_path, comparison_img)
        cv2.imwrite(os.path.join(stage7_frames_dir, f"comparison_frame_{idx:02d}_{filename}"), comparison_img)

        # Also save benchmark checkpoint samples in root output_dir
        if idx in [1, 5, 10, 20, 30]:
            cv2.imwrite(os.path.join(output_dir, f"eval_sample_{idx}_{filename}"), annotated)
            cv2.imwrite(os.path.join(output_dir, f"comparison_sample_{idx}_{filename}"), comparison_img)

        if idx % 5 == 0 or idx == len(image_paths):
            print(f"Processed frame {idx}/{len(image_paths)}... ({elapsed_ms:.1f}ms)", flush=True)

    # Calculate class-wise precision, recall, and F1
    metrics_by_class = {}
    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())

    print("\n-----------------------------------------------------------------")
    print("1. METRIK DETEKSI OBJEK APD (GROUND TRUTH VS PIPELINE)")
    print("-----------------------------------------------------------------")
    print(f"{'Kelas APD':<12} | {'GT':<6} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<8}")
    print("-" * 75)

    for cname in class_names:
        tp = tp_counts[cname]
        fp = fp_counts[cname]
        fn = fn_counts[cname]
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics_by_class[cname] = {
            "ground_truth": gt_counts[cname],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "recall": round(rec, 3),
            "precision": round(prec, 3),
            "f1_score": round(f1, 3),
        }
        print(f"{cname:<12} | {gt_counts[cname]:<6} | {tp:<6} | {fp:<6} | {fn:<6} | {rec * 100:>6.1f}% | {prec * 100:>8.1f}% | {f1:>8.3f}")

    global_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_f1 = (2 * global_prec * global_rec) / (global_prec + global_rec) if (global_prec + global_rec) > 0 else 0.0
    print("-" * 75)
    print(f"{'GLOBAL':<12} | {sum(gt_counts.values()):<6} | {total_tp:<6} | {total_fp:<6} | {total_fn:<6} | {global_rec * 100:>6.1f}% | {global_prec * 100:>8.1f}% | {global_f1:>8.3f}")

    print("\n-----------------------------------------------------------------")
    print("2. METRIK SCREENING PEKERJA (LOGIKA OVERLAP >= 80% PRD 10.6)")
    print("-----------------------------------------------------------------")
    print(f"Total Frame Diuji             : {len(image_paths)}")
    print(f"Total Pekerja Terdeteksi      : {total_workers_screened}")
    print(f"Pekerja Patuh Penuh (100% APD): {total_compliant_workers}")
    print(f"Pekerja Melanggar APD         : {total_non_compliant_workers}")
    overall_rate = (total_compliant_workers / total_workers_screened * 100.0) if total_workers_screened > 0 else 0.0
    print(f"Tingkat Kepatuhan Keseluruhan : {overall_rate:.2f}%")
    print("\nRekapitulasi Pelanggaran Komponen APD:")
    print(f"  - Helm Tidak Dipakai        : {helm_violations_total}")
    print(f"  - Sarung Tangan (Glove)     : {glove_violations_total}")
    print(f"  - Sepatu Safety             : {sepatu_violations_total}")
    print(f"  - Kacamata Safety           : {kacamata_violations_total}")

    mean_lat = float(np.mean(total_latency_ms))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    print("\n-----------------------------------------------------------------")
    print("3. PERFORMA & LATENSI INFERENSI")
    print("-----------------------------------------------------------------")
    print(f"Rata-rata Latensi Inferensi   : {mean_lat:.2f} ms / frame (FPS: {fps:.1f})")
    print(f"Latensi Tercepat / Terlama    : {min(total_latency_ms):.1f} ms / {max(total_latency_ms):.1f} ms")

    report_data = {
        "model_version": "YOLO11s APD PDU Tahap 7 Active Learning",
        "total_frames": len(image_paths),
        "total_workers": total_workers_screened,
        "compliant_workers": total_compliant_workers,
        "non_compliant_workers": total_non_compliant_workers,
        "compliance_rate": round(overall_rate, 2),
        "violations_breakdown": {
            "helm": helm_violations_total,
            "glove": glove_violations_total,
            "sepatu": sepatu_violations_total,
            "kacamata": kacamata_violations_total,
        },
        "class_metrics": metrics_by_class,
        "global_metrics": {
            "total_gt": sum(gt_counts.values()),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "recall": round(global_rec, 3),
            "precision": round(global_prec, 3),
            "f1_score": round(global_f1, 3),
        },
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "min_ms": round(float(min(total_latency_ms)), 2),
            "max_ms": round(float(max(total_latency_ms)), 2),
            "fps": round(fps, 1),
        },
    }

    report_path = os.path.join(output_dir, "test_set_evaluation_report.json")
    stage7_report_path = os.path.join(output_dir, "test_set_evaluation_report_stage7.json")
    for r_path in [report_path, stage7_report_path]:
        with open(r_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    print(f"\nLaporan evaluasi JSON tersimpan di:")
    print(f"  - {report_path}")
    print(f"  - {stage7_report_path}")

    # Komparasi dengan tahap sebelumnya jika tersedia
    previous_stages = [
        ("stage4", "Tahap 4 (Balanced)"),
        ("stage6", "Tahap 6 (Recovered)"),
        ("stage5", "Tahap 5 (Polished)"),
    ]
    for stage_key, stage_label in previous_stages:
        prev_report_path = os.path.join(output_dir, f"test_set_evaluation_report_{stage_key}.json")
        if os.path.exists(prev_report_path):
            try:
                with open(prev_report_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                prev_gm = prev_data.get("global_metrics", {})
                prev_cm = prev_data.get("class_metrics", {})
                print(f"\n=================================================================")
                print(f"  KOMPARASI HEAD-TO-HEAD: {stage_label} VS TAHAP 7 (ACTIVE LEARNING)")
                print(f"=================================================================")
                print(f"{'Metrik':<16} | {stage_label:<20} | {'Tahap 7 (Active)':<20} | {'Delta':<10}")
                print("-" * 75)
                
                p_prev, p7 = prev_gm.get("precision", 0.0), global_prec
                r_prev, r7 = prev_gm.get("recall", 0.0), global_rec
                f_prev, f7 = prev_gm.get("f1_score", 0.0), global_f1
                tp_prev, tp7 = prev_gm.get("total_tp", 0), total_tp
                fp_prev, fp7 = prev_gm.get("total_fp", 0), total_fp

                print(f"{'Precision':<16} | {p_prev * 100:>18.1f}% | {p7 * 100:>18.1f}% | {(p7 - p_prev) * 100:>+8.1f}%")
                print(f"{'Recall':<16} | {r_prev * 100:>18.1f}% | {r7 * 100:>18.1f}% | {(r7 - r_prev) * 100:>+8.1f}%")
                print(f"{'F1-Score':<16} | {f_prev:>19.3f} | {f7:>19.3f} | {(f7 - f_prev):>+9.3f}")
                print(f"{'True Positives':<16} | {tp_prev:>19} | {tp7:>19} | {(tp7 - tp_prev):>+9}")
                print(f"{'False Positives':<16} | {fp_prev:>19} | {fp7:>19} | {(fp7 - fp_prev):>+9}")
                print("-" * 75)
                print("Rincian Per Kelas (Precision / Recall / F1-Score):")
                for cname in class_names:
                    c_prev = prev_cm.get(cname, {})
                    c7 = metrics_by_class.get(cname, {})
                    print(f"  {cname.upper():<10} -> {stage_label}: P={c_prev.get('precision',0)*100:.1f}%, R={c_prev.get('recall',0)*100:.1f}%, F1={c_prev.get('f1_score',0):.3f} | Tahap 7: P={c7.get('precision',0)*100:.1f}%, R={c7.get('recall',0)*100:.1f}%, F1={c7.get('f1_score',0):.3f}")
                print("=================================================================\n")
            except Exception as e:
                print(f"Catatan: Komparasi {stage_label} dilewati ({e})")
            break  # Only compare against the most recent available previous stage


if __name__ == "__main__":
    evaluate_dataset()

