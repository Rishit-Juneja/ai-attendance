"""
Evaluate pipeline against ground truth.

Uses DirectArcFaceEmbedder (bypasses detection for pre-cropped faces).
Compares matched identities against ground truth XML.
"""
import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config, GALLERY_INDEX_PATH, GALLERY_META_PATH
from .data_loader import CASIALoader, DirectArcFaceEmbedder


@dataclass
class SceneResult:
    scene_name: str
    total_frames: int
    total_faces: int
    correct_matches: int = 0
    incorrect_matches: int = 0
    unknown_faces: int = 0
    ground_truth_matched: set = field(default_factory=set)
    per_person: dict = field(default_factory=dict)
    inference_ms_avg: float = 0.0


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = DirectArcFaceEmbedder()

        # Load FAISS gallery
        import faiss
        if GALLERY_INDEX_PATH.exists() and GALLERY_META_PATH.exists():
            self.index = faiss.read_index(str(GALLERY_INDEX_PATH))
            with open(GALLERY_META_PATH) as f:
                self.gallery_meta = json.load(f)
            print(f"[EVAL] Loaded gallery: {self.index.ntotal} faces")
        else:
            print("[ERROR] No gallery found. Run auto_enroll first.")
            self.index = None
            self.gallery_meta = []

    def _match(self, embedding: np.ndarray) -> tuple[str, str, float]:
        """Match embedding against gallery. Returns (name, roll, similarity)."""
        if self.index is None or self.index.ntotal == 0:
            return "Unknown", "", 0.0

        query = embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query, 1)
        score = float(similarities[0][0])
        idx = int(indices[0][0])

        if score > self.config.match_threshold and 0 <= idx < len(self.gallery_meta):
            m = self.gallery_meta[idx]
            return m["name"], m["roll"], score
        return "Unknown", "", score

    def evaluate_scene(self, scene_name: str, loader: CASIALoader, verbose=True) -> SceneResult | None:
        scene = loader.get_scene(scene_name)
        if not scene:
            print(f"[ERROR] Scene {scene_name} not found")
            return None

        # Build ground truth: frame_number → set of person_ids
        gt_by_frame: dict[int, set[str]] = defaultdict(set)
        for frame in scene.frames:
            gt_by_frame[frame.frame_number].add(frame.person_id)

        all_person_ids = set(f.person_id for f in scene.frames)

        result = SceneResult(
            scene_name=scene_name,
            total_frames=len(gt_by_frame),
            total_faces=len(scene.frames),
        )

        for pid in all_person_ids:
            result.per_person[pid] = {"correct": 0, "incorrect": 0, "total": 0}

        inference_times = []

        for face_frame in scene.frames:
            t0 = time.time()
            emb = self.embedder.embed(face_frame.image)
            inference_times.append((time.time() - t0) * 1000)

            if emb is None:
                result.unknown_faces += 1
                continue

            name, roll, score = self._match(emb)
            gt_persons = gt_by_frame.get(face_frame.frame_number, set())
            predicted_pid = roll if roll else None

            if predicted_pid and predicted_pid in gt_persons:
                result.correct_matches += 1
                result.ground_truth_matched.add(face_frame.frame_number)
                if predicted_pid in result.per_person:
                    result.per_person[predicted_pid]["correct"] += 1
                    result.per_person[predicted_pid]["total"] += 1
            elif predicted_pid:
                result.incorrect_matches += 1
                if predicted_pid in result.per_person:
                    result.per_person[predicted_pid]["incorrect"] += 1
                    result.per_person[predicted_pid]["total"] += 1
            else:
                result.unknown_faces += 1

        result.inference_ms_avg = np.mean(inference_times) if inference_times else 0

        if verbose:
            self._print_scene_result(result)

        return result

    def _print_scene_result(self, result: SceneResult):
        total_matched = result.correct_matches + result.incorrect_matches
        precision = result.correct_matches / max(total_matched, 1)
        recall = result.correct_matches / max(result.total_faces, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"\n{'='*60}")
        print(f"Scene: {result.scene_name}")
        print(f"  Faces: {result.total_faces}, Correct: {result.correct_matches}, "
              f"Incorrect: {result.incorrect_matches}, Unknown: {result.unknown_faces}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"  Avg inference: {result.inference_ms_avg:.1f}ms")
        print(f"{'='*60}")

    def evaluate_all(self, loader: CASIALoader, scenes: list[str] = None) -> dict:
        target_scenes = scenes or list(loader._scenes.keys())
        all_results = []

        for scene_name in sorted(target_scenes):
            result = self.evaluate_scene(scene_name, loader, verbose=True)
            if result:
                all_results.append(result)

        total_correct = sum(r.correct_matches for r in all_results)
        total_incorrect = sum(r.incorrect_matches for r in all_results)
        total_unknown = sum(r.unknown_faces for r in all_results)
        total_faces = sum(r.total_faces for r in all_results)
        total_matched = total_correct + total_incorrect

        overall = {
            "scenes_evaluated": len(all_results),
            "total_faces": total_faces,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "total_unknown": total_unknown,
            "precision": round(total_correct / max(total_matched, 1), 3),
            "recall": round(total_correct / max(total_faces, 1), 3),
            "f1": round(2 * total_correct / max(2 * total_correct + total_incorrect + total_unknown, 1), 3),
            "avg_inference_ms": round(np.mean([r.inference_ms_avg for r in all_results if r.inference_ms_avg > 0]), 1),
        }

        print(f"\n{'#'*60}")
        print(f"OVERALL ({overall['scenes_evaluated']} scenes)")
        print(f"  Faces: {overall['total_faces']}")
        print(f"  Precision: {overall['precision']}, Recall: {overall['recall']}, F1: {overall['f1']}")
        print(f"  Avg inference: {overall['avg_inference_ms']}ms")
        print(f"{'#'*60}")

        output_path = Path("data/reports/evaluation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(overall, f, indent=2)
        print(f"\n[SAVE] {output_path}")

        return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate attendance pipeline")
    parser.add_argument("--data-root", default="/tmp")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--gpu-profile", default="dev")
    parser.add_argument("--match-threshold", type=float, default=0.25)
    args = parser.parse_args()

    config = Config(
        gpu_profile=args.gpu_profile,
        match_threshold=args.match_threshold,
    )

    loader = CASIALoader(data_root=args.data_root)
    loader.discover()

    evaluator = Evaluator(config)
    evaluator.evaluate_all(loader, scenes=args.scenes)


if __name__ == "__main__":
    main()
