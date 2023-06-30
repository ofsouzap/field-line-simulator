from typing import Callable, Optional
from time import time as seconds_now

def millis_now(): return seconds_now() * 1000

class Timer:

    def __init__(self,
                 title: str,
                 log_callback: Optional[Callable[[str],None]] = None):

        self.title = title
        self.start_time = -1
        self.log_callback = log_callback or print

    def __enter__(self):

        self.start_time = millis_now()

    def __exit__(self, exc_type, exc_val, exc_tb):

        end_time = millis_now()

        assert self.start_time >= 0

        self.elapsed_time = end_time - self.start_time

        self.log_callback(f"Timer \"{self.title}\" took {self.elapsed_time:.0f}ms")