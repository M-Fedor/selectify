from io import IOBase
from threading import Lock
import struct


_lock = Lock()

def get_file_sort_id(file_name: str) -> int:
    file_name = file_name.split('.')[0]
    id = file_name.split('_')[-1]

    return int(id)


def sync_print(*a, **b) -> None:
    with _lock:
        print(*a, **b)


def write_binary(file: IOBase, data: any, length: int=0) -> None:
    if isinstance(data, int):
        file.write(data.to_bytes(length=length, byteorder='little', signed=False))
    if isinstance(data, str):
        file.write(data.encode('ascii'))


def read_binary(file: IOBase, length: int, data_type: str) -> any:
    data = file.read(length)

    if data_type == 'int':
        if not data:
            return -1
        return struct.unpack('<H', data)[0]
    if data_type == 'str':
        if not data:
            return ''
        return data.decode('ascii')


def get_length(data: any) -> int:
    if isinstance(data, int):
        return 2
    if isinstance(data, str):
        return len(data)
