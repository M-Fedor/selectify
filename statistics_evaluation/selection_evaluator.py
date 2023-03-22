import argparse
import csv
import sys
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from run_simulation.utils import read_binary
from statistics import Statistics


def parse_arguments() -> argparse.Namespace():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequencing-output', type=str, help='Output file produced by Virtual sequencer')
    parser.add_argument('--minimap-output', type=str, help='Output file produced by Minimap2')
    parser.add_argument('--produce_bed', action='store_true', default=False)

    args = parser.parse_args()

    if any(arg is None for arg in vars(args)):
        print('Incomplete inputs specified, nothing to do.')
        return None
    return args


def draw_histogram(data: dict) -> None:
    keys = sorted(data.keys())
    values = [data[k] for k in keys]

    plt.bar(keys, values)
    plt.show()


def load_sequencing_simulation_output(sequencing_output_path: str, container: Dict[str, int]) -> None:
    with open(sequencing_output_path, 'rb') as file:
        while True:
            read_id = read_binary(file, 36, 'str')
            sequenced_bases = read_binary(file, 4, 'int')

            if not read_id:
                break

            container[read_id] = sequenced_bases


def load_read_allignments(minimap_output_path: str, container: Dict[str, Tuple[str, int, int]]) -> None:
    with open(minimap_output_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for rows in csv_reader:
            read_id = rows[0]
            target = rows[5]
            target_begin = rows[7]
            target_end = rows[8]

            container[read_id] = (target, target_begin, target_end)


def sanity_check(
    simulated_read_lengths: Dict[str, int],
    read_allignments: Dict[str, Tuple[str, int, int]]
) -> None:
    pass


def get_statistics(
    simulated_read_lengths: Dict[str, int],
    read_allignments: Dict[str, Tuple[str, int, int]]
) -> Statistics:
    stats = Statistics()

    on_target_read_count = 0
    off_target_read_count = 0

    for read_id, read_length in simulated_read_lengths.items():
        on_target = True if read_id in read_allignments else False

        if on_target:
            stats.on_target_bases += read_length
            stats.on_target_read_length_distibution[read_length] += 1
            on_target_read_count += 1
        else:
            stats.off_target_bases += read_length
            stats.off_target_read_length_distibution[read_length] += 1
            off_target_read_count += 1

    stats.on_target_mean_read_length = stats.on_target_bases / on_target_read_count
    stats.off_target_mean_read_length = stats.off_target_bases/ off_target_read_count

    return stats


def visualize_statistics(stats: Statistics) -> None:
    print(len(stats.on_target_read_length_distibution.items()))
    print(len(stats.off_target_read_length_distibution.items()))

    draw_histogram(stats.on_target_read_length_distibution)
    draw_histogram(stats.off_target_read_length_distibution)

    print(f'On-target mean read length: {stats.on_target_mean_read_length}')
    print(f'Off-target mean read length: {stats.off_target_mean_read_length}')

    print(f'On-target bases: {stats.on_target_bases}')
    print(f'Off-target bases: {stats.off_target_bases}')


def process_data(args: argparse.Namespace) -> None:
    simulated_read_lengths = {}
    read_allignments = {}

    sanity_check(simulated_read_lengths, read_allignments)

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
