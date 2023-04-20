import argparse
import numpy as np
import os
import struct
from typing import Generator, List, Set, Tuple

from npy_append_array import NpyAppendArray
from ont_fast5_api.fast5_interface import get_fast5_file


REQUIRED_SIGNAL_LENGTH = 6_000
OUTPUT_SIGNAL_LENGTH = 5_000
OUTPUT_SIGNAL_CHUNKS = 3


def parse_arguments() -> argparse.Namespace():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mapped-reads-files', type=str, help='List of files containing read-ids of mapped reads')
    parser.add_argument('--extract-reads-files', type=str, help='List of files containing read-ids of mapped reads to be extracted')
    parser.add_argument('--fast5-dir', type=str, help='Directory containing .fast5 files')
    parser.add_argument('--output-dir', type=str, help='Directory containing extracted training data')

    args = parser.parse_args()

    if any([arg is None for arg in vars(args)]):
        print('Incomplete inputs specified, nothing to do.')
        return None
    return args


def med_mad(x: np.ndarray, factor: float=1.4826) -> Tuple[float, float]:
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def rescale_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad
    return np.clip(signal, -2.5, 2.5)


def parse_input_line(line: str) -> Tuple[str, int]:
    line_list = line.strip().split('\t')

    if len(line_list) == 1:
        return line_list + [0]
    
    file_name, threshold = line_list
    return file_name, int(threshold)


def get_read_ids(mapped_reads_path: str, read_ids: Set[str]) -> None:
    path = os.path.realpath(mapped_reads_path)
    mapped_reads_dir = os.path.dirname(path)
    remained_read_path = mapped_reads_path + '.remainder'

    file_count = 0

    with open(mapped_reads_path, 'r') as input_files, open(remained_read_path, 'w') as remained_files:
        for line in input_files:
            file_name, threshold = parse_input_line(line)
            path = os.path.join(mapped_reads_dir, file_name)

            with open(path, 'r') as read_file:
                file_count += 1

                read_count = 0
                for read_id in read_file:
                    read_count += 1
                    read_ids.add(read_id.strip())

                    if threshold > 0 and read_count == threshold:
                        break
                
                if threshold > 0 and read_count < threshold:
                    print(f"{file_name}\t{threshold - read_count}", file=remained_files)

                print(file_count, end='\r')


def extract_training_data(
    fast5_dir: str,
    positive_extract_read_ids: Set[str],
    positive_all_read_ids: Set[str]
) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
    positive_data = []
    negative_data = []
    negative_read_count = 0
    extracted_read_count = 0
    unsitable_positive_read_count = 0
    file_count = 0

    positive_extract_read_count = len(positive_extract_read_ids) * OUTPUT_SIGNAL_CHUNKS
    print(f'Positive reads to be extracted: {positive_extract_read_count}')

    for file_name in os.listdir(fast5_dir):  
        path = os.path.join(fast5_dir, file_name)
        file_count += 1

        with get_fast5_file(path, 'r') as f5_file:
            for read in f5_file.get_reads():
                read_id = read.get_read_id()

                if read_id in positive_extract_read_ids:
                    label = 1
                elif (
                    read_id not in positive_all_read_ids and
                    negative_read_count < positive_extract_read_count
                ):
                    label = 0
                else:
                    continue

                signal = read.get_raw_data()
                signal = rescale_signal(signal)

                for chunk_idx in range(OUTPUT_SIGNAL_CHUNKS):
                    chunk_start = chunk_idx * OUTPUT_SIGNAL_LENGTH
                    chunk_end = chunk_start + OUTPUT_SIGNAL_LENGTH

                    if len(signal) < chunk_end + 1_000:
                        if label:
                            unsitable_positive_read_count += 1
                        break

                    signal_chunk = signal[chunk_start : chunk_end]
                    labeled_signal = np.append(signal_chunk, label)
                    assert len(labeled_signal) == OUTPUT_SIGNAL_LENGTH + 1

                    if label:
                        extracted_read_count += 1
                        positive_data.append(labeled_signal)
                    else:
                        negative_read_count += 1
                        negative_data.append(labeled_signal)

            print(
                f"File count: {file_count}\t"
                f"Positive reads: {extracted_read_count}\t"
                f"Negative reads: {negative_read_count}\t"
                f"Unsuitable positive reads: {unsitable_positive_read_count}", end='\r'
            )

            if file_count % 200 == 0:
                yield (positive_data, negative_data)
                positive_data.clear()
                negative_data.clear()

            if (
                extracted_read_count == positive_extract_read_count and
                negative_read_count == positive_extract_read_count
            ):
                break

    yield (positive_data, negative_data)


def save_training_data(output_file, extracted_data: List[np.ndarray]) -> None:
    if extracted_data:
        data = np.array(extracted_data)
        data = data.astype(np.float32)
        output_file.append(data)


def permute_training_data(positive_path: str, negative_path: str, output_path: str) -> None:
    positive_data = np.load(positive_path, mmap_mode='r')
    negative_data = np.load(negative_path, mmap_mode='r')

    data = np.append(positive_data, negative_data, axis=0)
    np.random.shuffle(data)

    data = np.save(output_path, data)


def main() -> None:
    args = parse_arguments()
    if args is None:
        exit()

    positive_all_read_ids = set()
    positive_extract_read_ids = set()

    print('Reading mapped read-ids...')
    get_read_ids(args.mapped_reads_files, positive_all_read_ids)

    print('Reading mapped read-ids to be extracted...')
    get_read_ids(args.extract_reads_files, positive_extract_read_ids)

    print('Extracting training data...')
    positive_output_path = args.output_dir + '/training_data_positive.npy'
    negative_output_path = args.output_dir + '/training_data_negative.npy'

    with NpyAppendArray(positive_output_path) as positive_file, NpyAppendArray(negative_output_path) as negative_file:
        for extracted_data in extract_training_data(args.fast5_dir, positive_extract_read_ids, positive_all_read_ids):
            positive_data, negative_data = extracted_data
            save_training_data(positive_file, positive_data)
            save_training_data(negative_file, negative_data)

    print('\nGenerating random permutation...')
    joined_output_path = args.output_dir + '/training_data.npy'
    permute_training_data(positive_output_path, negative_output_path, joined_output_path)


if __name__ == "__main__":
    main()
