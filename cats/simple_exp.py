import torch
from torch.profiler import profile, record_function, ProfilerActivity

import cProfile

# import profile
import types


# def run(statement, filename=None, sort=-1):
#     prof = profile.Profile()
#     try:
#         result = prof.runcall(eval, statement, globals(), locals())
#     except SystemExit:
#         pass
#     if filename is not None:
#         prof.dump_stats(filename)
#     return prof


# if not hasattr(profile, "run"):
#     profile.run = types.MethodType(run, profile)

# Now try to import cProfile
import cProfile


# def simple_function():
#     return sum(range(1000000))
#
#
# cProfile.run("simple_function()")
def simple_function():
    return torch.ones(1000, 1000).sum()


with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function("simple_function"):
        simple_function()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
