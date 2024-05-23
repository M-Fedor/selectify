from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=False)
class Statistics:
    on_target_reads: int
    on_target_bases: int
    on_target_mean_read_length: float

    off_target_reads: int
    off_target_bases: int
    off_target_mean_read_length: float

    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int

    accuracy: float
    precission: float
    sensitivity: float
    specificity: float

    on_target_begins: Dict[int, int]
    on_target_unblocked_begins: Dict[int, int]
    on_target_proceeded_begins: Dict[int, int]

    on_target_read_length_distibution: Dict[int, int]
    off_target_read_length_distibution: Dict[int, int]

    def __init__(self) -> None:
        self.on_target_reads = 0
        self.on_target_bases = 0
        self.on_target_mean_read_length = 0
        self.off_target_reads = 0
        self.off_target_bases = 0
        self.off_target_mean_read_length = 0

        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

        self.accuracy = 0
        self.precission = 0
        self.sensitivity = 0
        self.specificity = 0

        self.on_target_begins = defaultdict(self.default_factory)
        self.on_target_unblocked_begins = defaultdict(self.default_factory)
        self.on_target_proceeded_begins = defaultdict(self.default_factory)

        self.on_target_read_length_distibution = defaultdict(self.default_factory)
        self.off_target_read_length_distibution = defaultdict(self.default_factory)


    def default_factory(self):
        return 0
