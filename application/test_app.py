from threading import Event
from time import sleep

from read_until import ReadCache

from run_simulation import ReadUntilSimulator


fast5_directory = 'fast5/'
sorted_directory = 'fast5_index/'

read_until_client = ReadUntilSimulator(
    fast5_read_directory=fast5_directory,
    sorted_read_directory=sorted_directory,
    split_read_interval=0.4,
    strand_type='dna',
    data_queue=ReadCache(512),
    one_chunk=True
)

read_until_client.run(0, 512)

try:
    Event().wait()
except KeyboardInterrupt:
    pass

read_until_client.reset(output_path='standard_sim.bin')
