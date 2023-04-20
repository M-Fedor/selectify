from threading import Event
from time import sleep

from read_until import ReadCache

from run_simulation import ReadUntilSimulator


fast5_directory = '/home/mfedor/zymoset/'
sorted_directory = '/home/mfedor/zymoset_index/'

read_until = ReadUntilSimulator(
    fast5_read_directory=fast5_directory,
    sorted_read_directory=sorted_directory,
    split_read_interval=0.4,
    strand_type='dna',
    data_queue=ReadCache(512),
    one_chunk=True
)

read_until.run(0, 512)

try:
    Event().wait()
except KeyboardInterrupt:
    pass

'''
while read_until.is_running:
    read_batch = read_until.get_read_chunks(batch_size=512)

    for channel, read in read_batch:
        sleep(0.001)
        read_until.unblock_read(channel, read.read_id)

    sleep(0.3)
'''

read_until.reset(output_path='zymo_standard_sim.bin')
