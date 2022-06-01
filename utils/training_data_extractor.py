import argparse
import numpy as np
import os

from ont_fast5_api.fast5_interface import get_fast5_file

REQUIRED_SIGNAL_LENGTH = 2500
OUTPUT_SIGNAL_LENGTH = 2000


def parse_arguments():
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


def med_mad(x, factor=1.4826):
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def rescale_signal(signal):
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad
    return np.clip(signal, -2.5, 2.5)


def parse_input_line(line):
    line_list = line.strip().split('\t')

    if len(line_list) == 1:
        return line_list + [0]
    
    file_name, threshold = line_list
    return file_name, int(threshold)


def test_read_count(file, treshold):
    count = 0
    for _ in file:
        count += 1
        if count == treshold:
            break

    return count


def get_read_ids(mapped_reads_file, result_set):
    file_count = 0

    path = os.path.realpath(mapped_reads_file)
    mapped_reads_dir = os.path.dirname(path)

    with open(mapped_reads_file, 'r') as input_files, open(mapped_reads_file + '.remainder', 'w') as remained_files:
        for line in input_files:
            file_name, threshold = parse_input_line(line)
            path = os.path.join(mapped_reads_dir, file_name)

            with open(path, mode="r") as read_file:
                file_count += 1
                
                if threshold > 0:
                    read_count = test_read_count(read_file, threshold)
                    if read_count < threshold:
                        print("%s\t%d" % (file_name, threshold - read_count), file=remained_files)

                read_count = 0
                for read_id in read_file:
                    read_count += 1
                    result_set.add(read_id.strip())

                    if threshold > 0 and read_count == threshold:
                        break

                print(file_count, end='\r')


def print_output(output_pos, output_neg):
    for read_id, label in read_id_labels.items():
        output_file = output_pos if label else output_neg

        for val in read_id_signals[read_id]:
            print("%s," % str(val), file=output_file, end='')
        print("%s" % str(label), file=output_file)

    read_id_labels.clear()
    read_id_signals.clear()


args = parse_arguments()
if args is None:
    exit()

positive_read_ids = set()
positive_training_read_ids = set()

print('Reading mapped read-ids...')
get_read_ids(args.mapped_reads_files, positive_read_ids)

print('Reading mapped read-ids to be extracted...')
get_read_ids(args.extract_reads_files, positive_training_read_ids)

positive_training_read_count = len(positive_training_read_ids)

print('Positive training read count: %d' % positive_training_read_count)

read_id_labels = {}
read_id_signals = {}
negative_read_count = 0
extracted_read_count = 0
file_count = 0

output_positive = args.output_dir + '/training_data_positive.txt'
output_negative = args.output_dir + '/training_data_negative.txt'

with open(output_positive, mode="w") as output_pos, open(output_negative, mode="w") as output_neg:
    for file_name in os.listdir(args.fast5_dir):
        path = os.path.join(args.fast5_dir, file_name)

        with get_fast5_file(path, mode="r") as f5_file:
            file_count += 1

            for read in f5_file.get_reads():
                read_id = read.get_read_id()

                if read_id in positive_training_read_ids:
                    label = 1
                elif (read_id not in positive_read_ids and
                      negative_read_count < positive_training_read_count
                ):
                    label = 0
                else:
                    continue

                signal = read.get_raw_data()

                if len(signal) < REQUIRED_SIGNAL_LENGTH:
                    continue

                signal = rescale_signal(signal)

                read_id_labels[read_id] = label
                read_id_signals[read_id] = signal[0:OUTPUT_SIGNAL_LENGTH]
                
                if label:
                    extracted_read_count += 1
                else:
                    negative_read_count += 1

            print("File count: %d\tNegative reads: %d\tPositive reads: %d" % 
                 (file_count, negative_read_count, extracted_read_count), end='\r'
            )

        if file_count % 100 == 0:
            print_output(output_pos, output_neg)

        if extracted_read_count == positive_training_read_count:
            break

    print_output(output_pos, output_neg)
