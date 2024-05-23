from __future__ import annotations

import heapq
import numpy as np
import os

from collections import deque
from copy import copy
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from threading import Event, Lock, Thread
from time import sleep, time_ns
from typing import Generator, List, Dict, Tuple

from .data import ReadSimulationData, LiveRead, LiveReadData, SimulatorEvent, SimulationStatistics
from .utils import get_file_sort_id, sync_print, read_binary, write_binary


MIN_READ_QUEUE_SIZE = 40
MAX_READ_QUEUE_SIZE = 100
EJECTION_SPEED = 40_000
SAMPLING_RATE_DNA = 4_000
SAMPLING_RATE_RNA = 3_012
BASES_CONVERSION_FACTOR_DNA = 4000 / 450
BASES_CONVERSION_FACTOR_RNA = 3012 / 70


class VirtualSequencer:

    def __init__(self,
        fast5_read_directory: str,
        sorted_read_directory: str,
        split_read_interval: float,
        strand_type: str='dna',
    ) -> None:
        self.fast5_read_directory = fast5_read_directory
        self.sorted_read_directory = sorted_read_directory
        self.split_read_interval = split_read_interval

        self.strand_type = strand_type
        self.sampling_rate = SAMPLING_RATE_RNA if strand_type == 'rna' else SAMPLING_RATE_DNA
        self.chunk_length = int(split_read_interval * self.sampling_rate)

        self.queue_lock = Lock()
        self.current_lock = Lock()
        self.live_lock = Lock()
        self.stat_lock = Lock()
        self.cancel_event = Event()
        self.preloader_wakeup_event = Event()
        self.provider_wakeup_event = SimulatorEvent()
        self.unblock_event = SimulatorEvent()
        self.live_read_event = Event()
        self.ready_event = Event()
        self.preloader_thread = None
        self.simulator_thread = None
        self.provider_thread = None

        self.read_index_files = {}
        self.sorted_fast5_files = []
        self.reset()


    def is_running(self) -> bool:
        return not self.cancel_event.is_set()


    def is_not_ready(self) -> bool:
        return not self.ready_event.is_set()


    def initialize(self) -> None:
        sync_print('Initializing virtual sequencer...')
        for file_name in os.listdir(self.sorted_read_directory):
            path = os.path.join(self.sorted_read_directory, file_name)
            file = open(path, 'rb')

            channel = read_binary(file, 2, 'int')
            assert channel not in self.read_index_files

            self.read_index_files[channel] = file
            self.read_queues[channel] = deque()
            self.current_reads[channel] = None
            self.live_reads[channel] = None

        for file_name in sorted(os.listdir(self.fast5_read_directory), key=get_file_sort_id):
            path = os.path.join(self.fast5_read_directory, file_name)
            self.sorted_fast5_files.append(get_fast5_file(path, mode='r'))

        self.preloader_thread = Thread(target=self._preload_reads, name='read_preloader')
        self.cancel_event.clear()
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

        for _, file in self.read_index_files.items():
            file.close()
        for file in self.sorted_fast5_files:
            file.close()

        self.preloader_wakeup_event.clear()
        self.provider_wakeup_event.clear()
        self.live_read_event.clear()
        self.unblock_event.clear()
        self.read_index_files = {}
        self.sorted_fast5_files = []
        self.read_queues = {}
        self.current_reads = {}
        self.live_reads = {}
        self.start_time = None
        self.statistics = SimulationStatistics()
        self.maximum_fast5_file_index = 0


    def start(self) -> None:
        sync_print('Start virtual sequencer...')
        self.start_time = _time()
        self.simulator_thread.start()
        self.provider_thread.start()


    def produce_output(self, output_path: str) -> None:
        with self.stat_lock, open(output_path, 'wb') as file:
            for read_id, (read_length, unblocked) in self.statistics.read_length_by_read_id.items():
                write_binary(file, read_id)
                write_binary(file, read_length, 4)
                write_binary(file, unblocked, 1)
        
        dir_name = os.path.dirname(output_path)
        with open(dir_name + '/additional_data.txt', 'w') as file:
            print(f'Maximum fast5 file index: {self.maximum_fast5_file_index}', file=file)


    def get_live_reads(self) -> Generator[List[LiveRead], None, None]:
        while self.is_running():
            live_reads = []

            self.live_read_event.wait()
            self.live_read_event.clear()

            with self.live_lock:
                for channel, read in self.live_reads.items():
                    if read is not None and len(read.raw_data) > 100:
                        live_reads.append(read)

            yield live_reads


    def stop_receiving(self, channel: int, read_id: str) -> None:
        with self.current_lock:
            assert channel in self.current_reads
            read = self.current_reads[channel]

        assert read.channel == channel

        if read.read_id == read_id:
            with self.current_lock:
                self.current_reads[channel].is_stopped = True


    def unblock(self, channel: int, read_id: str) -> None:
        with self.current_lock:
            assert channel in self.current_reads
            read = self.current_reads[channel]

        assert read.channel == channel

        if read.read_id != read_id or read.is_stopped:
            return
        
        with self.current_lock:
            self.current_reads[channel].is_stopped = True

        sequenced_signals = self._get_read_position(read.time_delta)
        if sequenced_signals >= len(read.raw_data) - 100:
            return

        saved_length = len(read.raw_data) - sequenced_signals
        saved_time = (saved_length / self.sampling_rate) - (sequenced_signals / EJECTION_SPEED)

        self.unblock_event.set((channel, saved_time, self._get_sequencing_time()))
        sequenced_bases = self._get_sequenced_bases(sequenced_signals)

        with self.stat_lock:
            assert read.read_id in self.statistics.read_length_by_read_id
            self.statistics.read_length_by_read_id[read_id] = (sequenced_bases, True)

        # sync_print(
        #     f'Unblocking read on channel {channel} read position {sequenced_bases} time-delta {read.time_delta} '
        #     f'saved length {saved_length} saved time {saved_time}'
        # )


    def _set_live_reads(self) -> None:
        sleep_time = self.split_read_interval
        active_channels = set()

        while self.is_running():
            while sleep_time > 0:
                t_start = _time()

                self.provider_wakeup_event.wait(sleep_time)
                self.provider_wakeup_event.clear()

                t_end = _time()
                sleep_time = sleep_time - t_end + t_start

            t_start = _time()

            for channel in self.provider_wakeup_event.get_data():
                active_channels.add(channel)

            for channel in active_channels.copy():
                with self.current_lock:
                    read = self.current_reads[channel]

                if read.read_id not in self.statistics.read_length_by_read_id:
                    with self.stat_lock:
                        sequenced_bases = self._get_sequenced_bases(len(read.raw_data))
                        self.statistics.read_length_by_read_id[read.read_id] = (sequenced_bases, False)

                chunk_end = self._get_read_position(read.time_delta)

                if chunk_end > len(read.raw_data) or read.is_stopped:
                    active_channels.remove(channel)
                    with self.live_lock:
                        self.live_reads[read.channel] = None
                else:
                    chunk_start = max(0, chunk_end - self.chunk_length)
                    signal_chunk = read.raw_data[chunk_start : chunk_end]

                    with self.live_lock:
                        self.live_reads[read.channel] = LiveRead(
                            channel=read.channel,
                            read_id=read.read_id,
                            number=read.number,
                            raw_data=signal_chunk
                        )

            self.live_read_event.set()

            t_end = _time()
            sleep_time = self.split_read_interval - t_end + t_start


    def _extract_fast5_read_data(
        self,
        channel: int,
        fast5_file_index: int,
        read_id: str
    ) -> ReadSimulationData:
        assert fast5_file_index >= 0 and fast5_file_index < len(self.sorted_fast5_files)

        fast5_file = self.sorted_fast5_files[fast5_file_index]
        read = Fast5Read(fast5_file, read_id)

        sampling_rate = read.handle['channel_id'].attrs['sampling_rate']
        read_number = read.handle['Raw'].attrs['read_number']
        start_time = read.handle['Raw'].attrs['start_time']
        raw_data = read.get_raw_data()

        time_delta = start_time / sampling_rate

        return ReadSimulationData(
            time_delta=time_delta,
            read_id=read_id,
            fast5_file_index=fast5_file_index,
            number=read_number,
            raw_data=raw_data,
            channel=channel,
            is_stopped =False,
        )


    def _preload_reads(self) -> None:
        while self.is_running():
            for channel, queue in self.read_queues.items():
                with self.queue_lock:
                    queue_length = len(queue)
                if queue_length > MIN_READ_QUEUE_SIZE + 10:
                    continue

                for _ in range(MAX_READ_QUEUE_SIZE - queue_length):
                    file = self.read_index_files[channel]

                    fast5_file_index = read_binary(file, 2, 'int')
                    read_id = read_binary(file, 36, 'str')

                    if not read_id:
                        break

                    self.maximum_fast5_file_index = max(fast5_file_index, self.maximum_fast5_file_index)
                    read = self._extract_fast5_read_data(channel, fast5_file_index, read_id)

                    with self.queue_lock:
                        if self.ready_event.is_set():
                            assert queue
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

        sync_print(f'Sequencing starts in {sorted_next_reads[0].time_delta} seconds...')
        sleep(sorted_next_reads[0].time_delta)

        while self.is_running():
            next_read = sorted_next_reads[0]
            next_time_delta = next_read.time_delta

            while next_read.time_delta == next_time_delta:
                if (next_read.read_id == next_reads[next_read.channel].read_id and
                    next_read.time_delta == next_reads[next_read.channel].time_delta
                ):
                    next_read = heapq.heappop(sorted_next_reads)

                    sequencing_time = self._get_sequencing_time()
                    internal_latency = sequencing_time - next_read.time_delta
                    next_read.time_delta = sequencing_time

                    with self.current_lock:
                        self.current_reads[next_read.channel] = next_read

                    self.statistics.saved_times[next_read.channel] -= internal_latency
                    self.provider_wakeup_event.set((next_read.channel))
                    self._set_next_read(next_read.channel, sorted_next_reads, next_reads)
                else:
                    heapq.heappop(sorted_next_reads)

                self.unblock_event.clear()
                self._unblock_reads(sorted_next_reads, next_reads)

                if not sorted_next_reads:
                    self.cancel_event.set()
                    return

                next_read = sorted_next_reads[0]

            sleep_time = next_read.time_delta - self._get_sequencing_time()

            while self.is_running() and sleep_time > 0:
                self.unblock_event.wait(sleep_time)
                self.unblock_event.clear()

                self._unblock_reads(sorted_next_reads, next_reads)
                sleep_time = sorted_next_reads[0].time_delta - self._get_sequencing_time()


    def _set_next_read(self,
        channel: int,
        sorted_reads: List[ReadSimulationData],
        reads: Dict[int, ReadSimulationData]
    ) -> None:
        queue = self.read_queues[channel]
        with self.queue_lock:
            if queue:
                simulation_read = queue.popleft()
                queue_length = len(queue)
            else:
                return

        saved_time = self.statistics.saved_times[channel]
        simulation_read.time_delta -= saved_time

        heapq.heappush(sorted_reads, simulation_read)
        reads[channel] = simulation_read

        assert simulation_read.time_delta >= self._get_sequencing_time()

        if queue_length <= MIN_READ_QUEUE_SIZE:
            self.preloader_wakeup_event.set()


    def _unblock_reads(self,
        sorted_reads: List[ReadSimulationData],
        reads: Dict[int, ReadSimulationData]
    ) -> None:
        for channel, saved_time, timestamp in self.unblock_event.get_data():
            internal_reaction_time = self._get_sequencing_time() - timestamp
            assert internal_reaction_time < 0.01

            read = copy(reads[channel])

            read.time_delta -= saved_time
            heapq.heappush(sorted_reads, read)
            reads[channel] = read

            assert read.time_delta >= timestamp
            self.statistics.saved_times[channel] += saved_time


    def _get_read_position(self, read_start: int) -> int:
        reading_time = self._get_sequencing_time() - read_start
        assert reading_time > 0

        return int(reading_time * self.sampling_rate)


    def _get_sequenced_bases(self, sequenced_signals: int) -> int:
        bases_conversion_factor = BASES_CONVERSION_FACTOR_RNA if self.strand_type == 'rna' else BASES_CONVERSION_FACTOR_DNA
        return int(sequenced_signals / bases_conversion_factor)


    def _get_sequencing_time(self) -> float:
        assert self.start_time is not None
        return _time() - self.start_time


def _time() -> float:
    return time_ns() / 1_000_000_000
