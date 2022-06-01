from read_until import ReadCache, ReadUntilClient
from threading import Event, Thread

from .virtual_sequencer import VirtualSequencer

from utils import sync_print


class ReadUntilSimulator(ReadUntilClient):

    def __init__(self, read_directory: str,
                chunk_time: float,
                cache_size: int,
                one_chunk: bool
    ) -> None:
        self.cache_size = cache_size
        self.one_chunk = one_chunk

        self.virtual_sequencer = VirtualSequencer(read_directory, chunk_time)
        self.data_queue = ReadCache(cache_size)

        self.running = Event()
        self.process_thread = None


    def run(self) -> None:
        sync_print('Starting Read Until API...')
        self.virtual_sequencer.initialize()
        self.virtual_sequencer.start()

        self.process_thread = Thread(target=self._process_reads, name='read_processor')
        self.running.set()
        self.process_thread.start()


    def reset(self) -> None:
        sync_print('Reseting Read Until API...')
        if self.process_thread is not None:
            self.running.clear()
            self.process_thread.join()

        self.virtual_sequencer.reset()

        self.data_queue = ReadCache(self.cache_size)


    @property
    def aquisition_progress(self):
        raise NotImplementedError


    def stop_receiving_read(self, read_channel: str, read_number: str) -> None:
        self.virtual_sequencer.stop_receiving(read_channel, read_number)


    def unblock_read(self, read_channel: str, read_number: str) -> None:
        self.virtual_sequencer.unblock(read_channel, read_number)


    def _process_reads(self) -> None:
        live_reads = self.virtual_sequencer.get_live_reads()

        for read_chunks in live_reads:
            if not self.is_running:
                break

            for chunk in read_chunks:
                self.data_queue[chunk.channel] = chunk
