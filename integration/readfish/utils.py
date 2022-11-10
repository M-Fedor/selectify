"""
These are some readfish utilities necessary for readfish
scripts to work authentically with minimal changes.
"""

from read_until_api_v2 import ReadCache, AccumulatingCache


def print_args(args, logger=None, exclude=None):
    """Print and format all arguments from the command line"""
    if exclude is None:
        exclude = []
    dirs = dir(args)
    m = max([len(a) for a in dirs if a[0] != '_'])
    for attr in dirs:
        if attr[0] != '_' and attr not in exclude and attr.lower() == attr:
            if logger is not None:
                logger.info("{a}={b}".format(a=attr, m=m, b=getattr(args, attr)))
            else:
                print('{a:<{m}}\t{b}'.format(a=attr, m=m, b=getattr(args, attr)))


def nice_join(seq, sep=", ", conjunction="or"):
    """Join lists nicely"""
    seq = [str(x) for x in seq]

    if len(seq) <= 1 or conjunction is None:
        return sep.join(seq)
    else:
        return "{} {} {}".format(sep.join(seq[:-1]), conjunction, seq[-1])


def get_cache_instance(cache_type: str, cache_size: int):
    if cache_type == 'ReadCache':
        return ReadCache(cache_size)
    if cache_type == 'AccumulatingCache':
        return AccumulatingCache(cache_size)
    
    raise NotImplementedError('Unknown cache type')
