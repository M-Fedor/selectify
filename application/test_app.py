from time import sleep

from run_simulation import ReadUntilSimulator


fast5_directory = 'selectify/data/'
sorted_directory = 'selectify/output'

read_until = ReadUntilSimulator(
    fast5_read_directory=fast5_directory,
    sorted_read_directory=sorted_directory,
    chunk_time=0.375,
    idelistic=False,
    cache_size = 512,
    one_chunk=True
)

read_until.run()

while read_until.is_running:
    read_batch = read_until.get_read_chunks(batch_size=512)

    for channel, read in read_batch:
        sleep(0.001)
        read_until.unblock_read(channel, read.number)

    sleep(0.1)

read_until.reset()
