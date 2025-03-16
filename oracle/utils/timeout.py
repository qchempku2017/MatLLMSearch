import functools
import signal
import os
import errno


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Timeout decorator, raises TimeoutError if function takes longer than specified.

    Args:
        seconds (int): Timeout in seconds.
        error_message (str): Error message to raise if timeout occurs.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError as e:
                raise e
            finally:
                signal.alarm(0)

        return wrapper

    return decorator