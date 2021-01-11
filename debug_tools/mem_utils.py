import resource

def ru():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1000000