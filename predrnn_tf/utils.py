def triplet(iterable):
    iters = [iter(iterable)] * 3
    return zip(*iters)
