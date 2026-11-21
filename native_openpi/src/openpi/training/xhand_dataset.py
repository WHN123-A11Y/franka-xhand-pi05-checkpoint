"""Dataset adapter for the local 19-D xHand LeRobot export.

The source export is kept in its original form.  Parquet is read directly with
PyArrow and images are loaded from the FFmpeg-generated per-episode cache.  This
avoids the incompatible Hugging Face ``List`` feature metadata in the pinned
LeRobot environment and keeps video decoding out of DataLoader workers.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import SupportsIndex

import numpy as np
import pyarrow.parquet as parquet


CAMERA_MAP = {
    "head_view": "base_0_rgb",
    "front_left_view": "left_wrist_0_rgb",
    "wrist_view": "right_wrist_0_rgb",
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_float32(table: parquet.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


class XHandDataset:
    """Random-access dataset with action chunks and cached RGB frames."""

    def __init__(
        self,
        data_root: str | Path,
        media_root: str | Path,
        action_horizon: int,
        *,
        max_cached_episodes: int = 1,
    ):
        self.data_root = Path(data_root)
        self.media_root = Path(media_root)
        self.action_horizon = action_horizon
        self.max_cached_episodes = max_cached_episodes
        self._media_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        info = json.loads((self.data_root / "meta/info.json").read_text())
        episodes = _read_jsonl(self.data_root / "meta/episodes.jsonl")
        task_rows = _read_jsonl(self.data_root / "meta/tasks.jsonl")
        self.tasks = self._load_tasks(task_rows)

        states = []
        actions = []
        task_indices = []
        episode_prompts = []
        lengths = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            path = self.data_root / "data" / f"chunk-{episode_index // int(info['chunks_size']):03d}" / (
                f"episode_{episode_index:06d}.parquet"
            )
            table = parquet.read_table(path, columns=["observation.state", "action", "task_index"])
            state = _to_float32(table, "observation.state")
            action = _to_float32(table, "action")
            task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
            expected_length = int(episode["length"])
            if state.shape != (expected_length, 19) or action.shape != (expected_length, 19):
                raise ValueError(
                    f"Episode {episode_index}: state={state.shape}, action={action.shape}, expected length={expected_length}"
                )
            if not np.all(task_index == task_index[0]):
                raise ValueError(f"Episode {episode_index} contains multiple task indices")
            states.append(state)
            actions.append(action)
            task_indices.append(int(task_index[0]))
            episode_tasks = episode.get("tasks", ())
            if isinstance(episode_tasks, str):
                episode_tasks = (episode_tasks,)
            prompt = next((str(task) for task in episode_tasks if str(task).strip()), None)
            if prompt is None:
                prompt = self.tasks[int(task_index[0])]
            episode_prompts.append(prompt)
            lengths.append(expected_length)

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.episode_ends = np.cumsum(self.lengths)
        self.episode_starts = np.concatenate([np.zeros(1, dtype=np.int64), self.episode_ends[:-1]])
        self.task_indices = np.asarray(task_indices, dtype=np.int64)
        self.episode_prompts = tuple(episode_prompts)

        if self.states.shape != (int(info["total_frames"]), 19):
            raise ValueError(f"Loaded state array has unexpected shape {self.states.shape}")
        if not self.media_root.is_dir():
            raise FileNotFoundError(f"FFmpeg media cache is missing: {self.media_root}")

    @staticmethod
    def _load_tasks(rows: list[dict]) -> dict[int, str]:
        result = {}
        for row_index, row in enumerate(rows):
            task_index = int(row.get("task_index", row_index))
            result[task_index] = str(row.get("task", row.get("instruction", row.get("description", ""))))
        if not result or any(not prompt for prompt in result.values()):
            raise ValueError(f"Could not read task prompts from tasks.jsonl: {rows}")
        return result

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def _episode_and_local_index(self, index: int) -> tuple[int, int]:
        episode = int(np.searchsorted(self.episode_ends, index, side="right"))
        local_index = index - int(self.episode_starts[episode])
        return episode, local_index

    def _load_media(self, episode: int) -> dict[str, np.ndarray]:
        if episode in self._media_cache:
            media = self._media_cache.pop(episode)
            self._media_cache[episode] = media
            return media

        mmap_path = self.media_root / f"episode_{episode:06d}"
        compressed_path = self.media_root / f"episode_{episode:06d}.npz"
        if mmap_path.is_dir():
            media = {
                suffix: np.load(mmap_path / f"{suffix}.npy", mmap_mode="r", allow_pickle=False)
                for suffix in CAMERA_MAP
            }
            path = mmap_path
        elif compressed_path.is_file():
            with np.load(compressed_path, allow_pickle=False) as archive:
                media = {key: archive[key] for key in archive.files}
            path = compressed_path
        else:
            raise FileNotFoundError(mmap_path)
        expected_length = int(self.lengths[episode])
        for suffix in CAMERA_MAP:
            if suffix not in media or media[suffix].shape[0] != expected_length:
                raise ValueError(f"Media cache {path} has invalid {suffix} shape")
        self._media_cache[episode] = media
        while len(self._media_cache) > self.max_cached_episodes:
            self._media_cache.popitem(last=False)
        return media

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        episode, local_index = self._episode_and_local_index(index)
        length = int(self.lengths[episode])
        frame_indices = np.minimum(np.arange(local_index, local_index + self.action_horizon), length - 1)
        action_start = int(self.episode_starts[episode])
        actions = self.actions[action_start + frame_indices].copy()
        padding = (np.arange(local_index, local_index + self.action_horizon) >= length).astype(bool)
        media = self._load_media(episode)

        image = {
            model_key: media[source_key][local_index]
            for source_key, model_key in CAMERA_MAP.items()
        }
        image_mask = {model_key: np.asarray(True) for model_key in image}
        return {
            "image": image,
            "image_mask": image_mask,
            "state": self.states[index].copy(),
            "actions": actions,
            "action_is_pad": padding,
            "prompt": np.asarray(self.episode_prompts[episode]),
        }
