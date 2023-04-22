import argparse
import numpy as np
from collections import OrderedDict
from time import sleep, time
from typing import Dict, List, Sequence

from arguments import parse_arguments
from classification import SarsCoV2Classifier, rescale_signal
from data import Decision, DecisionData
from run_simulation import LiveRead, ReadUntilSimulator
from integration.readfish import ReadCache


ON_TARGET_THRESHOLD = 0.8
OFF_TARGET_THRESHOLD = 0.8


def get_decision(prediction: Sequence[float]) -> Decision:
    on_target_confidence = prediction[0]
    off_target_confidence = prediction[1]

    if on_target_confidence > ON_TARGET_THRESHOLD:
        return (Decision.STOP_RECEIVING, 'stop-receiving')
    elif off_target_confidence > OFF_TARGET_THRESHOLD:
        return (Decision.UNBLOCK, 'unblock')
    return (Decision.PROCEED, 'proceed')


def insert(container: Dict[str, None], read_id: str, cache_size: int) -> None:
    if len(container) >= 4 * cache_size:
        container.popitem(last=False)
    container[read_id] = None


def process(
    classifier: SarsCoV2Classifier,
    read_batch: List[LiveRead],
    chunk_length: int
)-> List[DecisionData]:
    decisions = []
    raw_signals = []
    reduced_read_batch = []

    for channel, read in read_batch:
        if len(read.raw_data) > chunk_length:
            rescaled_signal = rescale_signal(read.raw_data[:chunk_length])
            raw_signals.append(rescaled_signal)
            reduced_read_batch.append((channel, read))

    if not reduced_read_batch:
        return []

    inputs = np.stack(raw_signals, axis=0)
    predictions = classifier.predict(inputs)

    for idx in range(len(reduced_read_batch)):
        channel, read = read_batch[idx]
        decision, decision_context = get_decision(predictions[idx])
        decisions.append(DecisionData(channel, read.read_id, decision, decision_context))
    
    return decisions


def selectify(args: argparse.Namespace) -> None:
    read_until_client = ReadUntilSimulator(
        fast5_read_directory=args.fast5_reads,
        sorted_read_directory=args.sorted_reads,
        split_read_interval=args.split_read_interval,
        strand_type='dna',
        data_queue=ReadCache(args.cache_size),
        one_chunk=False,
    )

    classifier = SarsCoV2Classifier()
    classifier.load(args.model)

    read_until_client.run(0, 512)

    proceeded_reads = OrderedDict()
    while read_until_client.is_running:
        t_start = time()

        read_batch = read_until_client.get_read_chunks(batch_size=512)
        decisions = process(classifier, read_batch, args.chunk_length)

        for decision in decisions:
            if decision.decision == Decision.UNBLOCK:
                read_until_client.unblock(decision.channel, decision.read_id)
            elif decision.decision == Decision.STOP_RECEIVING:
                read_until_client.stop_receiving_read(decision.channel, decision.read_id)
            else:
                if decision.read_id in proceeded_reads:
                    read_until_client.unblock(decision.channel, decision.read_id)
                    proceeded_reads.remove(decision.read_id)
                    decision.decision_context = 'unblock-max-chunks'
                else:
                    insert(proceeded_reads, decision.read_id, args.cache_size)
            
            if args.verbose:
                print(decision.decision_context)

        t_end = time()
        processing_time = t_end - t_start
        sleep_time = args.throttle - processing_time

        if read_batch:
            print(f'{len(read_batch)} Reads/{len(decisions)} Processed/{processing_time}s')

        if sleep_time > 0:
            sleep(sleep_time)

    read_until_client.reset(output_path=args.sequencing_output)


def main() -> None:
    args = parse_arguments()
    if args is None:
        exit()

    selectify(args)


if __name__ == "__main__":
    main()
