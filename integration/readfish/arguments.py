"""
These are readfish command-line arguments.

We added some new arguments to plug-in our software.
We did not remove any of the original arguments to avoid errors
while running our software using stored readfish commands.

Some of the original arguments are marked. They do not have any effect after 
ReadUntilSimulator integration.
"""

import argparse
import sys

from utils import nice_join


DEFAULT_WORKERS = 1
DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_CHANNELS = [1, 512]
DEFAULT_RUN_TIME = 172800
DEFAULT_UNBLOCK = 0.1
DEFAULT_READ_CACHE = "ReadCache"
DEFAULT_CACHE_SIZE = 512
DEFAULT_BATCH_SIZE = 512
DEFAULT_THROTTLE = 0.4
DEFAULT_SPLIT_READ_INTERVAL = 0.4

READ_CACHE_TYPES = ("ReadCache", "AccumulatingCache")
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")

BASE_ARGS = (
    # Following argument was added to integrate ReadUntilSimulator
    (
        "--fast5-reads",
        dict(
            metavar="FAST5-READS",
            help="Path to a directory containing .fast5 files",
            default=None,
        ),
    ),
    # Following argument was added to integrate ReadUntilSimulator
    (
        "--sorted-reads",
        dict(
            metavar="SORTED-READS",
            help="Path to a directory containig reads preprocessed by read_indexer.py",
            default=None,
        )
    ),
    # Following argument was added to integrate ReadUntilSimulator
    (
        "--split-read-interval",
        dict(
            metavar="SPLIT_READ_INTERVAL",
            type=float,
            help="Time duration after which an updated information about current read is"
                 "provided by ReadUntilSimulator",
            default=DEFAULT_SPLIT_READ_INTERVAL,
        )
    ),
    # Following argument has NO effect when bound with ReadUntilSimulator
    (
        "--host",
        dict(
            metavar="HOST",
            help="Argument has NO effect! Readfish is currently bound with ReadUntilSimulator",
        ),
    ),
    # Following argument has NO effect when bound with ReadUntilSimulator
    (
        "--port",
        dict(
            metavar="PORT",
            help="Argument has NO effect! Readfish is currently bound with ReadUntilSimulator",
        ),
    ),
    # Following argument has NO effect when bound with ReadUntilSimulator
    (
        "--device",
        dict(
            metavar="DEVICE",
            type=str,
            help="Argument has NO effect! Readfish is currently bound with ReadUntilSimulator",
        ),
    ),
    (
        "--experiment-name",
        dict(
            metavar="EXPERIMENT-NAME",
            type=str,
            help="Describe the experiment being run, enclose in quotes",
            required=True,
        ),
    ),
    (
        "--read-cache",
        dict(
            metavar="READ_CACHE",
            action="store",
            default=DEFAULT_READ_CACHE,
            choices=READ_CACHE_TYPES,
            help="One of: {} (default: {})".format(
                nice_join(READ_CACHE_TYPES), DEFAULT_READ_CACHE
            ),
        ),
    ),
      # TODO: delete workers
    (
        "--workers",
        dict(
            metavar="WORKERS",
            type=int,
            help="Number of worker threads (default: {})".format(DEFAULT_WORKERS),
            default=DEFAULT_WORKERS,
        ),
    ),
    (
        # ToDo: Delete and replace with api calls.
        "--channels",
        dict(
            metavar="CHANNELS",
            type=int,
            nargs=2,
            help="Channel range to use as a sequence, expects two integers "
                "separated by a space (default: {})".format(DEFAULT_CHANNELS),
            default=DEFAULT_CHANNELS,
        ),
    ),
    (
        "--run-time",
        dict(
            metavar="RUN-TIME",
            type=int,
            help="Period (seconds) to run the analysis (default: {:,})".format(
                DEFAULT_RUN_TIME
            ),
            default=DEFAULT_RUN_TIME,
        ),
    ),
    # Following argument has NO effect when bound with ReadUntilSimulator
    (
        "--unblock-duration",
        dict(
            metavar="UNBLOCK-DURATION",
            type=int,
            help="Argument has NO effect! Readfish is currently bound with ReadUntilSimulator",
            default=DEFAULT_UNBLOCK,
        ),
    ),
    (
        # ToDo:Deprecate so always the same size as the flowcell
        "--cache-size",
        dict(
            metavar="CACHE-SIZE",
            type=int,
            help="The size of the read cache in the ReadUntilClient (default: {:,})".format(
                DEFAULT_CACHE_SIZE
            ),
            default=DEFAULT_CACHE_SIZE,
        ),
    ),
    (
        # ToDo: Batch size should default to flowcell size unless otherwise specified.
        "--batch-size",
        dict(
            metavar="BATCH-SIZE",
            type=int,
            help="The maximum number of reads to pull from the read cache (default: {:,})".format(
                DEFAULT_BATCH_SIZE
            ),
            default=DEFAULT_BATCH_SIZE,
        ),
    ),
    (
        # ToDo: Determine if we need a minimum value for throttle which shouldn't be overridden.
        "--throttle",
        dict(
            metavar="THROTTLE",
            type=float,
            help="Time interval, in seconds, between requests to the ReadUntilClient (default: {})".format(
                DEFAULT_THROTTLE
            ),
            default=DEFAULT_THROTTLE,
        ),
    ),
    # Following argument has NO effect when bound with ReadUntilSimulator
    (
        "--dry-run",
        dict(
            action="store_true",
            help="Argument has NO effect! Readfish is currently bound with ReadUntilSimulator",
        ),
    ),
    (
        "--log-level",
        dict(
            metavar="LOG-LEVEL",
            action="store",
            default=DEFAULT_LOG_LEVEL,
            choices=LOG_LEVELS,
            help="One of: {}".format(nice_join(LOG_LEVELS)),
        ),
    ),
    (
        "--log-format",
        dict(
            metavar="LOG-FORMAT",
            action="store",
            default=DEFAULT_LOG_FORMAT,
            help="A standard Python logging format string (default: {!r})".format(
                DEFAULT_LOG_FORMAT.replace("%", "%%")
            ),
        ),
    ),
    (
        "--log-file",
        dict(
            metavar="LOG-FILE",
            action="store",
            default=None,
            help="A filename to write logs to, or None to write to the standard stream (default: None)",
        ),
    ),
)


def get_parser(extra_args=None, file=None, default_args=None):
    """Generic argument parser for ReadFish scripts

    Parameters
    ----------
    extra_args : Tuple[Tuple[str, dict], ...]
        Extra arguments to append onto the base arguments
    file : str
        Optional. __file__ from the python script, used for program string
    default_args : Tuple[Tuple[str, dict], ...]
        Arguments that form the base requirements for all ReadFish scripts

    Returns
    -------
    parser : argparse.ArgumentParser
        The argparse parser, used for raising parser errors manually
    arguments : argparse.ArgumentParser().parse_args()
        The parsed arguments
    """
    if default_args is None:
        args = BASE_ARGS
    else:
        args = default_args

    if extra_args is not None:
        args = args + extra_args

    if file is None:
        prog_string = "ReadFish API: {}".format(sys.argv[0].split("/")[-1])
    else:
        prog_string = "ReadFish API: {} ({})".format(sys.argv[0].split("/")[-1], file)

    parser = argparse.ArgumentParser(prog_string)
    for arg in args:
        flags = arg[0]
        if not isinstance(flags, tuple):
            flags = (flags,)
        parser.add_argument(*flags, **arg[1])

    return parser, parser.parse_args()
