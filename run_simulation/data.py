from __future__ import annotations

import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from threading import Event, Lock
from typing import Dict, List, Tuple


@dataclass(frozen=False)
class ReadData:
    time_delta: float
    read_id: str
    fast5_file_index: int

    def __lt__(self, other: ReadData) -> bool:
        return self.time_delta < other.time_delta


@dataclass(frozen=False)
class ReadSimulationData(ReadData):
    number: int
    raw_data: np.ndarray
    channel: int
    is_stopped: bool


@dataclass(frozen=False)
class LiveRead:
    channel: int
    read_id: str
    number: int
    raw_data: np.ndarray


@dataclass(frozen=False)
class LiveReadData:
    channel: int
    read_id: str
    number: int
    time_delta: float
    chunk_time_delta: float
    chunk_idx: int

    def __lt__(self, other: LiveReadData) -> bool:
        return self.chunk_time_delta < other.chunk_time_delta


class SimulatorEvent(Event):

    def __init__(self) -> None:
        super().__init__()
        self.data_lock = Lock()
        self.event_data = []


    def set(self, event_data: Tuple=None) -> None:
        super().set()

        if event_data is not None:
            with self.data_lock:
                self.event_data.append(event_data)


    def get_data(self) -> List[Tuple]:
        with self.data_lock:
            data = self.event_data[:]
            self.event_data.clear()
        return data


@dataclass(frozen=False)
class SimulationStatistics:
    read_length_by_read_id: Dict[str, Tuple[int, bool]]
    saved_times: Dict[int, float]


    def __init__(self):
        self.read_length_by_read_id = dict()
        self.saved_times = defaultdict(self.default_factory)


    def default_factory(self):
        return 0
