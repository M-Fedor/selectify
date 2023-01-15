"""
This script integrates ReadUntilSimulator together with VirtualSequencer
into the readfish's unblock_all script.

We made as few changes as possible. Just plugged-in our software while 
preserving all of the readfish inner workings and characteristics.

See the original at <https://github.com/LooseLab/readfish>.
"""

"""unblock_all.py

ReadUntil implementation that will only unblock reads. This should result in
a read length histogram that has very short peaks (~280-580bp) as these are the
smallest chunks that we can acquire. If you are not seeing these peaks, the
`split_reads_after_seconds` parameter in the configuration file may need to be
edited to 0.2-0.4.
"""
# Core imports
import functools
import logging
import sys
import time
from timeit import default_timer as timer

# Read Until Simulation imports
from arguments import BASE_ARGS
from utils import print_args, get_cache_instance
from run_simulation import read_until_simulator as read_until


_help = "Unblock all reads"
_cli = BASE_ARGS


def simple_analysis(
    client, duration, batch_size=512, throttle=0.4, unblock_duration=0.1
):
    """Analysis function

    Parameters
    ----------
    client : read_until_api.ReadUntilClient
        An instance of the ReadUntilClient object
    duration : int
        Time to run for, in seconds
    batch_size : int
        The number of reads to be retrieved from the ReadUntilClient at a time
    throttle : int or float
        The number of seconds interval between requests to the ReadUntilClient
    unblock_duration : int or float
        Time, in seconds, to apply unblock voltage

    Returns
    -------
    None
    """
    run_duration = time.time() + duration
    logger = logging.getLogger(__name__)

    while client.is_running and time.time() < run_duration:

        r = 0
        t0 = timer()
        unblock_batch_action_list = []
        stop_receiving_action_list = []
        for r, (channel, read) in enumerate(
                client.get_read_chunks(
                    batch_size=batch_size,
                    last=True,
                ),
                start=1,
        ):
            # Adding the channel and read.number to a list for a later batched unblock.
            unblock_batch_action_list.append((channel, read.read_id))
            stop_receiving_action_list.append((channel, read.read_id))

        if len(unblock_batch_action_list) > 0:
            client.unblock_read_batch(unblock_batch_action_list)
            client.stop_receiving_read_batch(stop_receiving_action_list)

        t1 = timer()
        if r:
            logger.info("Took {:.6f} for {} reads".format(t1 - t0, r))
        # limit the rate at which we make requests
        if t0 + throttle > t1:
            time.sleep(throttle + t0 - t1)
    else:
        logger.info("Finished analysis of reads as client stopped.")


def main():
    sys.exit(
        "This entry point is deprecated, please use 'readfish unblock-all' instead"
    )


def run(parser, args):
    # TODO: Move logging config to separate configuration file
    # TODO: use setup_logger here instead?
    # set up logging to file for DEBUG messages and above
    logging.basicConfig(
        level=logging.DEBUG,
        # TODO: args.log_format
        format="%(levelname)s %(asctime)s %(name)s %(message)s",
        filename=args.log_file,
        filemode="w",
    )

    # define a Handler that writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter(args.log_format)
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    # Start by logging sys.argv and the parameters used
    logger = logging.getLogger("Manager")
    logger.info(" ".join(sys.argv))
    print_args(args, logger=logger)

    read_until_client = read_until.ReadUntilSimulator(
        fast5_read_directory=args.fast5_reads,
        sorted_read_directory=args.sorted_reads,
        split_read_interval=args.split_read_interval,
        idealistic=False,
        data_queue=get_cache_instance("AccumulatingCache", args.cache_size),
        one_chunk=False,
    )

    read_until_client.run(
        first_channel=args.channels[0],
        last_channel=args.channels[-1],
    )

    try:
        simple_analysis(
        client=read_until_client,
        duration=args.run_time,
        batch_size=args.batch_size,
        throttle=args.throttle,
    )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(traceback.format_exc())
    finally:
        read_until_client.reset()


if __name__ == "__main__":
    main()
