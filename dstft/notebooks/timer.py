import numpy as np
import re
import itertools
import torch

# Raw data including "go" lines
raw_data = """
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0006 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0002 seconds
Elapsed time: 0.0017 seconds
Elapsed time: 0.0039 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0005 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0002 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0031 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0024 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0002 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0024 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0019 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0001 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0024 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0024 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0018 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0018 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0006 seconds
Elapsed time: 0.0024 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0002 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0021 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0020 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0023 seconds
go
Elapsed time: 0.0000 seconds
Elapsed time: 0.0001 seconds
Elapsed time: 0.0004 seconds
Elapsed time: 0.0022 seconds
"""

# Count the number of "go" blocks (i.e., time classes)
lines = raw_data.strip().splitlines()
num_classes = max(len(list(group)) for k, group in itertools.groupby(lines, lambda x: x.strip() == "go") if not k)


# Extract all elapsed time values
times = [float(match) for match in re.findall(r"Elapsed time: ([0-9.]+) seconds", raw_data)]

# Group times by position in each "go" block
grouped = list(zip(*[iter(times)]*num_classes))

# Compute averages and proportions per position
means = np.mean(grouped, axis=0)  #
medians = np.median(grouped, axis=0)
averages = means
total_mean = np.sum(averages)
percentages = 100*averages / total_mean

print(num_classes)
print(averages) 
print(total_mean)
print(percentages)
