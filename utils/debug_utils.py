from threading import Lock


lock = Lock()

def sync_print(*a, **b):
    with lock:
        print(*a, **b)
