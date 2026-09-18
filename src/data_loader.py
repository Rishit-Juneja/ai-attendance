"""
CASIA-SURF dataset loader.

Dataset structure on disk:
  /tmp/P1E_S1_C1/
    0001/          ← person ID
      00000100.pgm ← frame number (already a cropped 96x96 face)
    0003/
      ...

  /tmp/groundtruth/
    P1E_S1_C1.xml  ← lists frame numbers where faces appear

Naming: {Person}{Entry/Leave}_S{Scene}_C{Camera}
  P1E = Person 1, Entrance
  P1L = Person 1, Leaving

Since faces are already cropped (96x96 PGM), we bypass detection
and feed directly to ArcFace via get_feat().
"""
import glob
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceFrame:
    """A single face crop with metadata."""
    person_id: str          # e.g. "0015"
    frame_number: int       # e.g. 839
    image: np.ndarray       # BGR 96x96
    scene_name: str         # e.g. "P1E_S1_C1"
    entry_exit: str         # "E" or "L"


@dataclass
class SceneSequence:
    """All face frames for one camera scene, sorted by frame number."""
    scene_name: str
    person_type: str        # "P1E", "P1L", "P2E", "P2L"
    scene_num: int
    camera_num: int
    frames: list[FaceFrame]
    ground_truth_frames: set[int]


class CASIALoader:
    def __init__(self, data_root: str = "/tmp", ground_truth_dir: str = None):
        self.data_root = Path(data_root)
        self.gt_dir = Path(ground_truth_dir) if ground_truth_dir else self.data_root / "groundtruth"
        self._scenes: dict[str, SceneSequence] = {}

    def discover(self) -> dict[str, SceneSequence]:
        scene_pattern = re.compile(r"^(P[12][EL])_S(\d+)_C(\d+)$")
        scenes = {}

        for entry in sorted(self.data_root.iterdir()):
            if not entry.is_dir():
                continue
            m = scene_pattern.match(entry.name)
            if not m:
                continue

            person_type = m.group(1)
            scene_num = int(m.group(2))
            camera_num = int(m.group(3))

            frames = []
            for person_dir in sorted(entry.iterdir()):
                if not person_dir.is_dir():
                    continue
                person_id = person_dir.name
                for pgm_path in sorted(person_dir.glob("*.pgm")):
                    frame_num = int(pgm_path.stem)
                    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    frames.append(FaceFrame(
                        person_id=person_id,
                        frame_number=frame_num,
                        image=img_bgr,
                        scene_name=entry.name,
                        entry_exit=person_type[-1],
                    ))

            frames.sort(key=lambda f: f.frame_number)
            gt_frames = self._load_ground_truth(entry.name)

            scenes[entry.name] = SceneSequence(
                scene_name=entry.name,
                person_type=person_type,
                scene_num=scene_num,
                camera_num=camera_num,
                frames=frames,
                ground_truth_frames=gt_frames,
            )

        self._scenes = scenes
        print(f"[DATA] Discovered {len(scenes)} scenes, {sum(len(s.frames) for s in scenes.values())} total face frames")
        return scenes

    def _load_ground_truth(self, scene_name: str) -> set[int]:
        xml_path = self.gt_dir / f"{scene_name}.xml"
        if not xml_path.exists():
            return set()
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        frames = set()
        for frame_elem in root.findall("frame"):
            num = frame_elem.get("number")
            if num:
                frames.add(int(num))
        return frames

    def get_scene(self, scene_name: str) -> SceneSequence | None:
        return self._scenes.get(scene_name)

    def iter_scenes(self):
        return self._scenes.values()

    def get_person_gallery_images(self, scene_name: str, max_per_person: int = 1) -> dict[str, list[np.ndarray]]:
        """For enrollment: pick the sharpest face crop per person."""
        scene = self._scenes.get(scene_name)
        if not scene:
            return {}

        by_person: dict[str, list[tuple[float, np.ndarray]]] = {}
        for frame in scene.frames:
            if frame.person_id not in by_person:
                by_person[frame.person_id] = []
            lap_var = cv2.Laplacian(frame.image, cv2.CV_64F).var()
            by_person[frame.person_id].append((lap_var, frame.image))

        gallery = {}
        for pid, candidates in by_person.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            gallery[pid] = [img for _, img in candidates[:max_per_person]]

        return gallery

    def iter_frames_at_fps(self, scene_name: str, fps: int = 3):
        """Yield FaceFrames in temporal order at given fps."""
        scene = self._scenes.get(scene_name)
        if not scene:
            return

        frame_interval = 1.0 / fps
        frame_idx = 0
        for face_frame in scene.frames:
            yield face_frame, frame_idx * frame_interval, frame_idx
            frame_idx += 1


class DirectArcFaceEmbedder:
    """
    Embed pre-cropped faces directly using ArcFace, bypassing detection.
    For use with CASIA dataset (96x96 PGM crops).
    """

    def __init__(self, model_path: str = None, input_size: int = 112):
        from insightface.model_zoo import get_model
        if model_path is None:
            model_path = str(Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx")
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=0)
        self.input_size = input_size

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """
        Embed a single pre-cropped face image (any size, will be resized).
        Returns L2-normalized 512-dim embedding or None.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        resized = cv2.resize(face_bgr, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_CUBIC)
        feat = self.model.get_feat(resized).flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat.astype(np.float32)

    def embed_batch(self, faces: list[np.ndarray]) -> list[np.ndarray | None]:
        """Embed multiple faces."""
        return [self.embed(f) for f in faces]


def get_all_person_ids(scenes: dict[str, SceneSequence]) -> set[str]:
    """Get unique person IDs across all scenes."""
    ids = set()
    for scene in scenes.values():
        for frame in scene.frames:
            ids.add(frame.person_id)
    return ids
