from numpy import int16
from pathlib import Path
from read_until import ReadUntilClient
from threading import Event, Thread
from typing import List, Tuple

from .data import LiveRead
from .virtual_sequencer import VirtualSequencer
from .statistics import draw_histogram, write_file
from .utils import sync_print


class ReadUntilSimulator(ReadUntilClient):

    def __init__(self,
        fast5_read_directory: str,
        sorted_read_directory: str,
        split_read_interval: float,
        idealistic: bool,
        data_queue,
        one_chunk: bool,
    ) -> None:
        self.one_chunk = one_chunk
        self.data_queue = data_queue
        self.signal_dtype = int16

        self.virtual_sequencer = VirtualSequencer(
            fast5_read_directory,
            sorted_read_directory,
            split_read_interval,
            idealistic
        )

        self.running = Event()
        self.process_thread = None

        # Readfish compatibility
        self.mk_run_dir = Path('.')


    def run(self, first_channel: int, last_channel: int) -> None:
        sync_print('Start Read Until API...')
        self.virtual_sequencer.initialize()
        self.virtual_sequencer.start()

        self.process_thread = Thread(
            target=self._process_reads, 
            args=(first_channel, last_channel),
            name='read_processor'
        )

        self.running.set()
        self.process_thread.start()


    def reset(self, data_queue=None, produce_stats: bool=False) -> None:
        sync_print('Reset Read Until API...')
        if self.process_thread is not None:
            self.running.clear()
            self.process_thread.join()

        self.statistics = self.virtual_sequencer.get_statistics()
        self.virtual_sequencer.reset()
        self.data_queue = data_queue

        if produce_stats:
            self.produce_statistics()


    def produce_statistics(self) -> None:
        draw_histogram(self.statistics.read_length_distribution)
        write_file(self.statistics.read_length_by_read_id, 'stats/read_id_sequenced_lenghts.bin')


    @property
    def aquisition_progress(self) -> None:
        raise NotImplementedError

    def get_read_chunks(self, batch_size: int=1, last: bool=True) -> List[LiveRead]:
        return self.data_queue.popitems(batch_size, last)

    def stop_receiving_read(self, read_channel: int, read_number: str) -> None:
        self.virtual_sequencer.stop_receiving(read_channel, read_number)

    def stop_receiving_read_batch(self, identifier_list: List[Tuple[int, str]]) -> None:
        for identifier in identifier_list:
            read_channel, read_number = identifier
            self.stop_receiving_read(read_channel, read_number)

    def unblock_read(self, read_channel: int, read_number: str) -> None:
        self.virtual_sequencer.unblock(read_channel, read_number)

    def unblock_read_batch(self, identifier_list: List[Tuple[int, str]]) -> None:
        for identifier in identifier_list:
            read_channel, read_number = identifier
            self.unblock_read(read_channel, read_number)


    def _process_reads(self, first_channel: int, last_channel: int) -> None:
        live_reads = self.virtual_sequencer.get_live_reads()

        while self.is_running and self.virtual_sequencer.is_not_canceled():
            for read_chunks in live_reads:
                for chunk in read_chunks:
                    if first_channel <= chunk.channel and chunk.channel <= last_channel:
                        self.data_queue[chunk.channel] = chunk

        self.running.clear()
