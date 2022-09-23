from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from threading import Event, Lock
from typing import List, Tuple


@dataclass(frozen=False)
class ReadData:
    time_delta: float
    read_id: str
    fast5_file_index: int

    def __lt__(self, other: ReadData) -> bool:
        return self.time_delta < other.time_delta


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
