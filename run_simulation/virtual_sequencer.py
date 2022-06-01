import heapq
import numpy as np
import os

from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import sleep, time_ns
from typing import Generator, List, Dict, Tuple

from .data import ReadData, SimulatorEvent
from utils import sync_print


MIN_READ_QUEUE_SIZE = 40
MAX_READ_QUEUE_SIZE = 100
EJECTION_SPEED = 8_000
SEQUENCING_SPEED = 4_000


class VirtualSequencer:

    @dataclass(frozen=True)
    class LiveRead:
        channel: str
        number: str
        signal: np.ndarray


    @dataclass(frozen=False)
    class LiveReadData:
        channel: str
        read_id: str
        time_delta: float
        chunk_time_delta: float
        chunk_idx: int

        def __lt__(self, other) -> bool:
            return self.chunk_time_delta < other.chunk_time_delta


    @dataclass(frozen=False)
    class ReadSimulationData(ReadData):
        channel: str
        is_stopped: bool


    def __init__(self, read_directory: str, chunk_time: float) -> None:
        self.read_directory = read_directory
        self.chunk_time = chunk_time
        self.chunk_length = chunk_time * SEQUENCING_SPEED

        self.queue_lock = Lock()
        self.current_lock = Lock()
        self.live_lock = Lock()
        self.cancel_event = Event()
        self.preloader_wakeup_event = Event()
        self.provider_wakeup_event = SimulatorEvent()
        self.unblock_event = SimulatorEvent()
        self.live_read_event = Event()
        self.ready_event = Event()
        self.preloader_thread = None
        self.simulator_thread = None
        self.provider_thread = None

        self.read_files = {}
        self.reset()


    def is_not_canceled(self) -> bool:
        return not self.cancel_event.is_set()


    def is_not_ready(self) -> bool:
        return not self.ready_event.is_set()


    def initialize(self) -> None:
        sync_print('Initialize virtual sequencer...')
        for file_name in os.listdir(self.read_directory):
            path = os.path.join(self.read_directory, file_name)
            file = open(path, 'r')

            channel = file.readline().strip()
            assert channel not in self.read_files

            self.read_files[channel] = file
            self.read_queues[channel] = deque()
            self.current_reads[channel] = None
            self.live_reads[channel] = None
            self.saved_times[channel] = 0

        self.preloader_thread = Thread(target=self._preload_reads, name='read_preloader')
        self.preloader_thread.start()

        self.simulator_thread = Thread(target=self._simulate, name='read_simulator')
        self.provider_thread = Thread(target=self._set_live_reads, name='live_read_provider')

        self.ready_event.wait()


    def reset(self) -> None:
        sync_print('Reset virtual sequencer...')
        self.cancel_event.set()
        self.preloader_wakeup_event.set()
        self.provider_wakeup_event.set()
        self.live_read_event.set()
        self.unblock_event.set()
        self.ready_event.clear()

        if self.preloader_thread is not None:
            assert self.simulator_thread is not None
            self.preloader_thread.join()
            self.simulator_thread.join()
            self.provider_thread.join()

        for _, file in self.read_files.items():
            file.close()

        self.preloader_wakeup_event.clear()
        self.provider_wakeup_event.clear()
        self.live_read_event.clear()
        self.unblock_event.clear()
        self.cancel_event.clear()
        self.read_files = {}
        self.read_queues = {}
        self.current_reads = {}
        self.live_reads = {}
        self.saved_times = {}
        self.start_time = None


    def start(self) -> None:
        sync_print('Start virtual sequencer...')
        self.cancel_event.clear()
        self.simulator_thread.start()
        self.provider_thread.start()


    def get_live_reads(self) -> Generator[List[LiveRead], None, None]:
        while self.is_not_canceled():
            live_reads = []

            self.live_read_event.wait()
            self.live_read_event.clear()

            with self.live_lock:
                for channel, read in self.live_reads.items():
                    if read is not None:
                        live_reads.append(read)
                        self.live_reads[channel] = None

            yield live_reads


    def stop_receiving(self, channel: str, read_id: str) -> None:
        with self.current_lock:
            assert channel in self.current_reads

            read = self.current_reads[channel]
            assert read.channel == channel

            if read.read_id == read_id:
                self.current_reads[channel].is_stopped = True


    def unblock(self, channel: str, read_id: str) -> None:
        with self.current_lock:
            assert channel in self.current_reads

            read = self.current_reads[channel]
            assert read.channel == channel

            if read.read_id == read_id:
                self.current_reads[channel].is_stopped = True
            else:
                return

        read_pos = _get_read_position(self.start_time, read.time_delta)
        if read_pos >= len(read.signal):
            return

        saved_length = len(read.signal) - read_pos
        saved_time = (saved_length / SEQUENCING_SPEED) - (read_pos / EJECTION_SPEED)

        sync_print("Unblocking read on channel %s read position %d time-delta %.2f saved length %d saved time %.2f" % 
                    (channel,
                    read_pos,
                    read.time_delta,
                    saved_length,
                    saved_time
                )
        )

        self.unblock_event.set((channel, saved_time))


    def _set_live_reads(self) -> None:
        sorted_next_channels = []
        sleep_time = None

        while self.is_not_canceled():
            while sleep_time is None or sleep_time > 0:
                self.provider_wakeup_event.wait(sleep_time)
                self.provider_wakeup_event.clear()

                for channel in self.provider_wakeup_event.get_data():
                    with self.current_lock:
                        read = self.current_reads[channel]

                    time_delta = read.time_delta
                    heapq.heappush(sorted_next_channels, self.LiveReadData(
                            channel=channel,
                            read_id=read.read_id,
                            time_delta=time_delta,
                            chunk_time_delta=time_delta + self.chunk_time,
                            chunk_idx=1
                        )
                    )

                read_data = sorted_next_channels[0]
                sleep_time = read_data.chunk_time_delta - _get_sequencing_time(self.start_time)

            read_data = heapq.heappop(sorted_next_channels)

            with self.current_lock:
                read = self.current_reads[read_data.channel]
            chunk_start, chunk_end = _get_chunk_positions(read_data.chunk_idx, self.chunk_length)
    
            if (chunk_end > len(read.signal) or
                read.is_stopped or
                read.read_id != read_data.read_id or
                read.time_delta != read_data.time_delta
            ):
                with self.live_lock:
                    self.live_reads[read.channel] = None
            else:
                assert _get_sequencing_time(self.start_time) - read.time_delta >= read_data.chunk_idx * self.chunk_time

                signal_chunk = read.signal[chunk_start : chunk_end]

                with self.live_lock:
                    self.live_reads[read.channel] = self.LiveRead(
                        channel=read.channel,
                        number=read.read_id,
                        signal=signal_chunk
                    )

                read_data.chunk_time_delta += self.chunk_time
                read_data.chunk_idx += 1
                heapq.heappush(sorted_next_channels, read_data)
                self.live_read_event.set()

            read_data = sorted_next_channels[0]
            sleep_time = read_data.chunk_time_delta - _get_sequencing_time(self.start_time)


    def _preload_reads(self) -> None:
        while self.is_not_canceled():
            for channel, queue in self.read_queues.items():
                queue_length = len(queue)

                if queue_length > MIN_READ_QUEUE_SIZE:
                    continue
                for _ in range(MAX_READ_QUEUE_SIZE - queue_length):
                    read_line = self.read_files[channel].readline()

                    if not read_line:
                        break

                    read_line = read_line.strip().split(',')
                    read = ReadData(
                        time_delta=int(read_line[0]),
                        read_id=read_line[1],
                        signal=np.asfarray(read_line[2:-1])
                    )

                    with self.queue_lock:
                        queue.append(read)

            if self.is_not_ready():
                self.ready_event.set()

            self.preloader_wakeup_event.wait()
            self.preloader_wakeup_event.clear()


    def _simulate(self) -> None:
        sorted_next_reads = []
        next_reads = {}

        for channel in self.read_queues.keys():
            self._set_next_read(channel, sorted_next_reads, next_reads)

        if not sorted_next_reads:
            self.cancel_event.set()
            return

        self.start_time = _time()
        sleep(sorted_next_reads[0].time_delta)

        while self.is_not_canceled():
            next_read = sorted_next_reads[0]
            next_time_delta = next_read.time_delta

            while next_read.time_delta == next_time_delta:
                if (next_read.read_id == next_reads[next_read.channel].read_id and
                    next_read.time_delta == next_reads[next_read.channel].time_delta
                ):
                    with self.current_lock:
                        next_read = heapq.heappop(sorted_next_reads)
                        next_read.time_delta = _get_sequencing_time(self.start_time)
                        self.current_reads[next_read.channel] = next_read
                        self.provider_wakeup_event.set((next_read.channel))

                    self._set_next_read(next_read.channel, sorted_next_reads, next_reads)
                else:
                    heapq.heappop(sorted_next_reads)

                if not sorted_next_reads:
                    self.cancel_event.set()
                    return

                next_read = sorted_next_reads[0]

            sleep_time = next_read.time_delta - _get_sequencing_time(self.start_time)

            while self.is_not_canceled() and sleep_time > 0:
                self.unblock_event.wait(sleep_time)
                self.unblock_event.clear()

                self._unblock_reads(sorted_next_reads, next_reads)
                sleep_time = sorted_next_reads[0].time_delta - _get_sequencing_time(self.start_time)


    def _set_next_read(self, channel: str,
        sorted_reads: List[ReadSimulationData],
        reads: Dict[str, ReadSimulationData]
    ) -> None:
        queue = self.read_queues[channel]

        with self.queue_lock:
            if queue:
                read = queue.popleft()
            else:
                return
        
        simulation_read = self.ReadSimulationData(
            time_delta=self._get_read_time_delta(channel, read),
            read_id=read.read_id,
            signal=read.signal,
            channel=channel,
            is_stopped =False,
        )

        heapq.heappush(sorted_reads, simulation_read)
        reads[channel] = simulation_read

        if len(queue) <= MIN_READ_QUEUE_SIZE:
            self.preloader_wakeup_event.set()


    def _unblock_reads(self,
        sorted_reads: List[ReadSimulationData],
        reads: Dict[str, ReadSimulationData]
    ) -> None:
        for channel, saved_time in self.unblock_event.get_data():
            self.saved_times[channel] += saved_time
            reads[channel].time_delta = self._get_read_time_delta(channel, reads[channel])
            heapq.heappush(sorted_reads, reads[channel])


    def _get_read_time_delta(self, channel: str, read: ReadData) -> float:
        saved_time = self.saved_times[channel]
        return read.time_delta - saved_time


def _get_sequencing_time(seq_start: float) -> float:
    return _time() - seq_start


def _get_chunk_positions(
    chunk_idx: int,
    chunk_length: float
) -> Tuple[int, int]:
    chunk_end = (chunk_idx) * chunk_length
    return int(chunk_end - chunk_length), int(chunk_end)


def _get_read_position(seq_start: float, read_start: int) -> int:
    reading_time = _get_sequencing_time(seq_start) - read_start
    assert reading_time > 0

    return int(reading_time * SEQUENCING_SPEED)


def _time() -> float:
    return time_ns() / 1_000_000_000
