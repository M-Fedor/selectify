import argparse
import os

from collections import deque
from io import IOBase
from ont_fast5_api.fast5_interface import get_fast5_file
from typing import Dict

from data import ReadData
from utils import get_file_sort_id, write_binary

QUEUE_MINIMAL_SIZE = 500
OUTPUT_CACHE_SIZE = 100
OUTPUT_INTERVAL = 10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fast5-reads', type=str, help='Directory containing .fast5 files')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed reads')

    args = parser.parse_args()

    if args.fast5_reads is None or args.output_dir is None:
        print('Incomplete inputs specified, nothing to do.')
        return None
    return args


def insert_read(channel: int, read: ReadData, container: Dict[int, str]) -> None:
    if channel not in container:
        container[channel] = deque([read])
        return

    channel_queue = container[channel]
    sorting_distance = 0

    for idx in range(len(channel_queue), 0, -1):
        if channel_queue[idx - 1] < read:
            channel_queue.insert(idx, read)
            break
        elif idx - 1 == 0:
            channel_queue.insert(0, read)
        sorting_distance += 1

    assert sorting_distance < QUEUE_MINIMAL_SIZE


def sorting_safe_quard(container: Dict[int, str]) -> None:
    for _, queue in container.items():
        for idx in range(len(queue) - 1):
            assert queue[idx] < queue[idx + 1]


def output_reads(
    container: Dict[int, str],
    files: Dict[int, IOBase],
    output_dir: str,
    last_batch: bool=False
) -> None:
    for channel, queue in container.items():
        if channel not in files:
            files[channel] = open(output_dir + '/sorted_reads_' + str(channel), 'wb')
            write_binary(files[channel], int(channel), 2)

        file = files[channel]
        queue_size = 0 if last_batch else QUEUE_MINIMAL_SIZE

        for _ in range(len(queue) - queue_size):
            read = queue.popleft()

            write_binary(file, read.fast5_file_index, 2)
            write_binary(file, read.read_id)


def process_reads(args: argparse.Namespace) -> None:
    reads_by_channel_number = {}
    files_by_channel_number = {}
    file_idx = 0

    for file_name in sorted(os.listdir(args.fast5_reads), key=get_file_sort_id):
        path = os.path.join(args.fast5_reads, file_name)

        if not os.path.isfile(path):
            continue

        with get_fast5_file(path, mode="r") as f5_file:
            for read in f5_file.get_reads():
                channel_number = read.handle['channel_id'].attrs['channel_number'].decode('utf-8')
                sampling_rate = read.handle['channel_id'].attrs['sampling_rate']
                start_time = read.handle['Raw'].attrs['start_time']
                read_id = read.read_id

                time_delta = start_time / sampling_rate

                read = ReadData(time_delta, read_id, file_idx)
                insert_read(channel_number, read, reads_by_channel_number)

            file_idx += 1               

        if file_idx % OUTPUT_INTERVAL == 0:
            print('Done: ' + str(file_idx), end='\r')
        if file_idx % OUTPUT_CACHE_SIZE == 0:
            # sorting_safe_quard(reads_by_channel_number)
            output_reads(reads_by_channel_number, files_by_channel_number, args.output_dir)

    output_reads(reads_by_channel_number, files_by_channel_number, args.output_dir, last_batch=True)

    for _, file in files_by_channel_number.items():
        file.close()


def main() -> None:
    args = parse_arguments()
    if args is None:
        exit()

    process_reads(args)


if __name__ == "__main__":
    main()
