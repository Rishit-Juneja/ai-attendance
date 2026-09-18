"""
Auto-enrollment from CASIA dataset.

Uses DirectArcFaceEmbedder to bypass detection (faces are already cropped).
Picks the sharpest face per person, embeds, stores in FAISS gallery.
"""
import argparse
import json
from pathlib import Path

import cv2
import faiss
import numpy as np

from .config import ENROLLMENT_DIR, GALLERY_DIR, DEFAULT_CONFIG
from .data_loader import CASIALoader, DirectArcFaceEmbedder


GALLERY_INDEX_PATH = GALLERY_DIR / "gallery.index"
GALLERY_META_PATH = GALLERY_DIR / "gallery_meta.json"


def auto_enroll(
    loader: CASIALoader,
    embedder: DirectArcFaceEmbedder,
    source_scenes: list[str] = None,
    max_per_person: int = 3,
):
    GALLERY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Collect best faces per person across scenes
    person_images: dict[str, list[tuple[float, np.ndarray, str]]] = {}

    scenes_to_use = source_scenes or list(loader._scenes.keys())
    for scene_name in scenes_to_use:
        scene = loader.get_scene(scene_name)
        if not scene:
            continue
        gallery = loader.get_person_gallery_images(scene_name, max_per_person=max_per_person)
        for pid, imgs in gallery.items():
            if pid not in person_images:
                person_images[pid] = []
            for img in imgs:
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                person_images[pid].append((lap_var, img, scene_name))

    if not person_images:
        print("[ERROR] No persons found in source scenes")
        return

    print(f"[ENROLL] Found {len(person_images)} unique persons across {len(scenes_to_use)} scenes")

    dim = 512
    index = faiss.IndexFlatIP(dim)
    meta = []

    enrolled = 0
    for pid in sorted(person_images.keys()):
        candidates = person_images[pid]
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sharpness, best_img, best_scene = candidates[0]

        emb = embedder.embed(best_img)
        if emb is None:
            print(f"  [SKIP] {pid}: embedding failed")
            continue

        index.add(emb.reshape(1, -1))
        meta.append({
            "name": f"Person_{pid}",
            "roll": pid,
            "photo": f"auto_enrolled_from_{best_scene}",
            "embedding_idx": len(meta),
            "sharpness": round(best_sharpness, 1),
            "source_scene": best_scene,
        })

        save_path = ENROLLMENT_DIR / f"{pid}_best.jpg"
        cv2.imwrite(str(save_path), best_img)

        enrolled += 1
        print(f"  [OK] {pid} (sharpness={best_sharpness:.0f}, from {best_scene})")

    faiss.write_index(index, str(GALLERY_INDEX_PATH))
    with open(GALLERY_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[ENROLL] Done: {enrolled}/{len(person_images)} persons enrolled")
    print(f"  Gallery: {GALLERY_INDEX_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Auto-enroll from CASIA dataset")
    parser.add_argument("--data-root", default="/tmp")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes to enroll from")
    parser.add_argument("--max-per-person", type=int, default=3)
    args = parser.parse_args()

    loader = CASIALoader(data_root=args.data_root)
    loader.discover()

    embedder = DirectArcFaceEmbedder()

    auto_enroll(
        loader=loader,
        embedder=embedder,
        source_scenes=args.scenes,
        max_per_person=args.max_per_person,
    )


if __name__ == "__main__":
    main()
