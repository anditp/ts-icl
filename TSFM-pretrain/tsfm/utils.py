import random
import timeit

import petname


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def generate_run_name(seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)
    # two words, remove separator or use '-' if you prefer
    words = petname.generate(2, separator="-")  # e.g. "ancient-fox"
    # ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{words}"
