from threading import Lock


_lock = Lock()

def get_file_sort_id(file_name: str) -> int:
    file_name = file_name.split('.')[0]
    id = file_name.split('_')[-1]

    return int(id)


def sync_print(*a, **b) -> None:
    with _lock:
        print(*a, **b)
