"""
These are utilities carved out from read_until_api_v2
needed for ReadUntilSimulator integration into readfish.
"""

import time
import logging
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool


def run_workflow(client, partial_analysis_func, n_workers, run_time, runner_kwargs=None):
    """Run an analysis function against a ReadUntilClient

    Parameters
    ----------
    client : read_until.ReadUntilClient
        An instance of the ReadUntilClient object
    partial_analysis_func : partial function
        Analysis function to process reads, should
        exit when client.is_running == False
    n_workers : int
        Number of analysis worker functions to run
    run_time : int
        Time, in seconds, to run the analysis for
    runner_kwargs : dict
        Keyword arguments to pass to client.run()

    Returns
    -------
    list
        Results from the analysis function, one item per worker

    """
    if runner_kwargs is None:
        runner_kwargs = dict()

    logger = logging.getLogger("Manager")

    results = []
    pool = ThreadPool(n_workers)
    logger.info("Creating {} workers".format(n_workers))
    try:
        # start the client
        client.run(runner_kwargs["first_channel"], runner_kwargs["last_channel"])
        # start a pool of workers
        for _ in range(n_workers):
            results.append(pool.apply_async(partial_analysis_func))
        pool.close()
        # wait a bit before closing down
        time.sleep(run_time)
        logger.info("Sending reset")
        client.reset()
        pool.join()
    except KeyboardInterrupt:
        logger.info("Caught ctrl-c, terminating workflow.")
        client.reset()
    except Exception:
        client.reset()
        raise

    # collect results (if any)
    collected = []
    for result in results:
        try:
            res = result.get(5)
        except TimeoutError:
            logger.warning("Worker function did not exit successfully.")
            # collected.append(None)
        except Exception as e:
            logger.exception("EXCEPT", exc_info=e)
            # logger.warning("Worker raise exception: {}".format(repr(e)))
        else:
            logger.info("Worker exited successfully.")
            collected.append(res)
    pool.terminate()
    return collected
