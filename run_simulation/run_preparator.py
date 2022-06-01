import argparse
import os

from collections import deque
from ont_fast5_api.fast5_interface import get_fast5_file

from data import ReadData

QUEUE_MINIMAL_SIZE = 500
OUTPUT_CACHE_SIZE = 200


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fast5-reads', type=str, help='Directory containing .fast5 files')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed reads')

    args = parser.parse_args()

    if args.fast5_reads is None or args.output_dir is None:
        print('Incomplete inputs specified, nothing to do.')
        return None
    return args


def insert_read(channel: int, read: ReadData, container: map):
    if channel not in container:
        container[channel] = deque([read])
        out_of_order_counts[channel] = 0
        max_sorting_distance[channel] = 0
        return

    channel_queue = container[channel]

    if channel_queue and channel_queue[-1].time_delta > read.time_delta:
        out_of_order_counts[channel] += 1

    sorting_distance = 0

    for idx in range(len(channel_queue), 0, -1):
        if channel_queue[idx - 1].time_delta < read.time_delta:
            max_sorting_distance[channel] = len(channel_queue) - idx
            channel_queue.insert(idx, read)
            break
        elif idx - 1 == 0:
            max_sorting_distance[channel] = len(channel_queue)
            channel_queue.insert(0, read)
        sorting_distance += 1

    assert sorting_distance < QUEUE_MINIMAL_SIZE


def sorting_safe_quard(container: map):
    for _, queue in container.items():
        for idx in range(len(queue) - 1):
            assert queue[idx].time_delta < queue[idx + 1].time_delta


def output_reads(container: map, files: map, output_dir: str, end: bool = False):
    for channel, queue in container.items():
        if channel not in files:
            files[channel] = open(output_dir + '/sorted_reads_' + str(channel) + '.txt', 'w')
            print(channel, file=files[channel])

        file = files[channel]
        queue_size = 0 if end else QUEUE_MINIMAL_SIZE
        for _ in range(len(queue) - queue_size):
            read = queue.popleft()
            print("%d,%s" % (read.time_delta, read.read_id), file=file, end='')

            for val in read.signal:
                print(',%d' % val, file=file, end='')
            print('', file=file)


def get_file_sort_id(file_name):
    file_name = file_name.split('.')[0]
    id = file_name.split('_')[-1]
    return int(id)


args = parse_arguments()
if args is None:
    exit()

reads_by_channel_id = {}
files_by_channel_id = {}
out_of_order_counts = {}
max_sorting_distance = {}
file_count = 0

for file_name in sorted(os.listdir(args.fast5_reads), key=get_file_sort_id):
    path = os.path.join(args.fast5_reads, file_name)
    if not os.path.isfile(path):
        continue

    with get_fast5_file(path, mode="r") as f5_file:
        file_count += 1

        for read in f5_file.get_reads():
            read_id = read.read_id
            channel_number = read.handle[read.global_key + 'channel_id'].attrs['channel_number'].decode('utf-8')
            start_time = read.handle['Raw'].attrs['start_time']
            sampling_rate = read.handle[read.global_key + 'channel_id'].attrs['sampling_rate']
            signal = read.get_raw_data()

            time_delta = start_time / sampling_rate
            read_data = ReadData(read_id, time_delta, signal)
            insert_read(channel_number, read_data, reads_by_channel_id)

    if file_count % OUTPUT_CACHE_SIZE == 0:
        print(file_count, end='\r')
        sorting_safe_quard(reads_by_channel_id)
        output_reads(reads_by_channel_id, files_by_channel_id, args.output_dir)

output_reads(reads_by_channel_id, files_by_channel_id, args.output_dir, True)

for _, file in files_by_channel_id.items():
    file.close()

with open('sorting_stats.txt', 'w') as f:
    for channel in out_of_order_counts.keys():
        print("%d %d" % (out_of_order_counts[channel], max_sorting_distance[channel]), file=f)
