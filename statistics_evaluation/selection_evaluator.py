import argparse
import csv
import sys
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from run_simulation.utils import read_binary
from statistics import Statistics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequencing-output', type=str, help='Output file produced by Virtual sequencer')
    parser.add_argument('--minimap-output', type=str, help='Output file produced by Minimap2')
    parser.add_argument('--produce_bed', action='store_true', default=False)

    args = parser.parse_args()

    if any(arg is None for arg in vars(args)):
        print('Incomplete inputs specified, nothing to do.')
        return None
    return args


def draw_histogram(
    data: dict,
    x_label: str=None,
    y_label: str=None,
    title: str=None
) -> None:
    keys = sorted(data.keys())
    values = [data[k] for k in keys]

    plt.bar(keys, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    '''
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    '''

    plt.show()
    plt.figure(figsize=(10,10))


def get_accuracy(stats: Statistics) -> float:
    try:
        return (stats.true_positives + stats.true_negatives) / (stats.false_positives + stats.false_negatives + stats.true_positives + stats.true_negatives)
    except ZeroDivisionError:
        return 0


def get_sensitivity(stats: Statistics) -> float:
    try:
        return stats.true_positives / (stats.true_positives + stats.false_negatives)
    except ZeroDivisionError:
        return 0


def get_specificity(stats: Statistics) -> float:
    try:
        return stats.true_negatives / (stats.true_negatives + stats.false_positives)
    except ZeroDivisionError:
        return 0


def get_precision(stats: Statistics) -> float:
    try:
        return stats.true_positives / (stats.true_positives + stats.false_positives)
    except ZeroDivisionError:
        return 0


def load_sequencing_simulation_output(sequencing_output_path: str, container: Dict[str, int]) -> None:
    with open(sequencing_output_path, 'rb') as file:
        while True:
            read_id = read_binary(file, 36, 'str')
            sequenced_bases = read_binary(file, 4, 'int')
            unblocked = read_binary(file, 1, 'int')

            if not read_id:
                break

            container[read_id] = (sequenced_bases, unblocked)


def load_read_allignments(minimap_output_path: str, container: Dict[str, Tuple[str, int, int]]) -> None:
    with open(minimap_output_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for rows in csv_reader:
            read_id = rows[0]
            target = rows[5]
            target_begin = rows[7]
            target_end = rows[8]

            container[read_id] = (target, target_begin, target_end)


def get_statistics(
    simulated_read_lengths: Dict[str, int],
    read_allignments: Dict[str, Tuple[str, int, int]]
) -> Statistics:
    stats = Statistics()

    for read_id, (read_length, unblocked) in simulated_read_lengths.items():
        on_target = True if read_id in read_allignments else False

        if on_target:
            stats.on_target_bases += read_length
            stats.on_target_read_length_distibution[read_length] += 1
            stats.on_target_reads += 1

            _, target_begin, _ = read_allignments[read_id]

            if unblocked:
                stats.on_target_unblocked_begins[target_begin] += 1
                stats.false_negatives += 1
            else:
                stats.on_target_proceeded_begins[target_begin] += 1
                stats.true_positives += 1
            stats.on_target_begins[target_begin] += 1
        else:
            stats.off_target_bases += read_length
            stats.off_target_read_length_distibution[read_length] += 1
            stats.off_target_reads += 1

            if unblocked:
                stats.true_negatives += 1
            else:
                stats.false_positives += 1

    for begin in stats.on_target_begins.keys():
        stats.on_target_unblocked_begins[begin] /= stats.on_target_begins[begin]
        stats.on_target_proceeded_begins[begin] /= stats.on_target_begins[begin]

    stats.accuracy = get_accuracy(stats)
    stats.sensitivity = get_sensitivity(stats)
    stats.specificity = get_specificity(stats)
    stats.precision = get_precision(stats)

    stats.on_target_mean_read_length = stats.on_target_bases / stats.on_target_reads
    stats.off_target_mean_read_length = stats.off_target_bases/ stats.off_target_reads

    return stats


def visualize_statistics(stats: Statistics) -> None:
    draw_histogram(stats.on_target_read_length_distibution, 'Read length', 'Number of reads', 'On-target Read Length Distribution')
    draw_histogram(stats.off_target_read_length_distibution,'Read length', 'Number of reads', 'Off-target Read Length Distribution')

    draw_histogram(stats.on_target_begins)
    draw_histogram(stats.on_target_unblocked_begins, 'Read alignment starting position', 'Fraction of reads', 'On-target Read Alignment Distribution')
    draw_histogram(stats.on_target_proceeded_begins, 'Read alignment starting position', 'Fraction of reads', 'On-target Read Alignment Distribution')

    print(f'On-target read count: {stats.on_target_reads}')
    print(f'Off-target read count: {stats.off_target_reads}')

    print(f'On-target mean read length: {stats.on_target_mean_read_length}')
    print(f'Off-target mean read length: {stats.off_target_mean_read_length}')

    print(f'On-target bases: {stats.on_target_bases}')
    print(f'Off-target bases: {stats.off_target_bases}')

    print(f'Accuracy: {stats.accuracy}')
    print(f'Sensitivity: {stats.sensitivity}')
    print(f'Specificity: {stats.specificity}')
    print(f'Precission: {stats.precision}')


def process_data(args: argparse.Namespace) -> None:
    simulated_read_lengths = {}
    read_allignments = {}

    load_sequencing_simulation_output(args.sequencing_output, simulated_read_lengths)
    load_read_allignments(args.minimap_output, read_allignments)

    stats = get_statistics(simulated_read_lengths, read_allignments)

    # if args.produce_bed:
    #     produce_bed(simulated_read_lengths, read_allignments)        

    visualize_statistics(stats)


def main() -> None:
    args = parse_arguments()
    if args is None:
        exit()

    process_data(args)


if __name__ == "__main__":
    main()
