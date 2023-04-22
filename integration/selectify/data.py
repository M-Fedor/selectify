from enum import Enum
from dataclasses import dataclass


class Decision(Enum):
    UNBLOCK = 1
    STOP_RECEIVING = 2
    PROCEED = 3


@dataclass(frozen=False)
class DecisionData():
    channel: int
    read_id: str
    decision: Decision
    decision_context: str
