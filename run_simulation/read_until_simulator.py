from read_until import ReadCache, ReadUntilClient
from threading import Event, Thread

from .virtual_sequencer import VirtualSequencer

from .utils import sync_print


class ReadUntilSimulator(ReadUntilClient):

    def __init__(self,
        fast5_read_directory: str,
        sorted_read_directory: str,
        split_read_interval: float,
        idealistic: bool,
        data_queue,
        one_chunk: bool
    ) -> None:
        self.one_chunk = one_chunk
        self.data_queue = data_queue

        self.virtual_sequencer = VirtualSequencer(
            fast5_read_directory,
            sorted_read_directory,
            split_read_interval,
            idealistic
        )

        self.running = Event()
        self.process_thread = None


    def run(self, first_channel, last_channel) -> None:
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


    def reset(self, data_queue=None) -> None:
        sync_print('Reset Read Until API...')
        if self.process_thread is not None:
            self.running.clear()
            self.process_thread.join()

        self.virtual_sequencer.reset()

        self.data_queue = data_queue


    @property
    def aquisition_progress(self) -> None:
        raise NotImplementedError

    def get_read_chunks(self, batch_size=1, last=True):
        return self.data_queue.popitems(batch_size, last=last)

    def stop_receiving_read(self, read_channel: str, read_number: str) -> None:
        self.virtual_sequencer.stop_receiving(read_channel, read_number)


    def unblock_read(self, read_channel: str, read_number: str) -> None:
        self.virtual_sequencer.unblock(read_channel, read_number)


    def _process_reads(self, first_channel, last_channel) -> None:
        live_reads = self.virtual_sequencer.get_live_reads()

        while self.is_running and self.virtual_sequencer.is_not_canceled():
            for read_chunks in live_reads:
                for chunk in read_chunks:
                    channel_number = int(chunk.channel)

                    if first_channel <= channel_number and channel_number <= last_channel:
                        self.data_queue[chunk.channel] = chunk

        self.running.clear()
