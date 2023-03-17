from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=False)
class Statistics:
    on_target_bases: int
    on_target_mean_read_length: float

    off_target_bases: int
    off_target_mean_read_length: float

    on_target_read_length_distibution: Dict[int, int]
    off_target_read_length_distibution: Dict[int, int]

    def __init__(self) -> None:
        self.on_target_bases = 0
        self.on_target_mean_read_length = 0
        self.off_target_bases = 0
        self.off_target_mean_read_length = 0
        self.on_target_read_length_distibution = defaultdict(self.default_factory)
        self.off_target_read_length_distibution = defaultdict(self.default_factory)


    def default_factory(self):
        return 0
