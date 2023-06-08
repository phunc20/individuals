import time


def function2():
    return list(range(10_000))


def function():
    return list(range(10_000))


if __name__ == "__main__":
    start = time.perf_counter()
    function()
    end = time.perf_counter()
    elapsed = end - start
    print(f'{function.__name__}  took {elapsed * 10**6:.2f} ms.')

    start = time.perf_counter()
    function2()
    end = time.perf_counter()
    elapsed = end - start
    print(f'{function2.__name__} took {elapsed * 10**6:.2f} ms.')

