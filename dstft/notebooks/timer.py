import numpy as np
import re
import itertools
import torch

x = np.array([2.33333333e-05, 1.13333333e-04, 4.73333333e-04, 2.16666667e-03])  # 0.0006666666666666666
print(x/x.sum())
# Raw data including "go" lines
raw_data = """
go
Elapsed time: 0.0289 seconds
Elapsed time: 0.0297 seconds
go
Elapsed time: 0.0294 seconds
Elapsed time: 0.0284 seconds
go
Elapsed time: 0.0281 seconds
Elapsed time: 0.0259 seconds
go
Elapsed time: 0.0279 seconds
Elapsed time: 0.0263 seconds
go
Elapsed time: 0.0282 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0300 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0306 seconds
Elapsed time: 0.0296 seconds
go
Elapsed time: 0.0297 seconds
Elapsed time: 0.0306 seconds
go
Elapsed time: 0.0511 seconds
Elapsed time: 0.0477 seconds
go
Elapsed time: 0.0346 seconds
Elapsed time: 0.0316 seconds
go
Elapsed time: 0.0338 seconds
Elapsed time: 0.0324 seconds
go
Elapsed time: 0.0483 seconds
Elapsed time: 0.0498 seconds
go
Elapsed time: 0.0314 seconds
Elapsed time: 0.0306 seconds
go
Elapsed time: 0.0316 seconds
Elapsed time: 0.0338 seconds
go
Elapsed time: 0.0355 seconds
Elapsed time: 0.0309 seconds
go
Elapsed time: 0.0389 seconds
Elapsed time: 0.0326 seconds
go
Elapsed time: 0.0334 seconds
Elapsed time: 0.0299 seconds
go
Elapsed time: 0.0324 seconds
Elapsed time: 0.0283 seconds
go
Elapsed time: 0.0312 seconds
Elapsed time: 0.0272 seconds
go
Elapsed time: 0.0314 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0307 seconds
Elapsed time: 0.0270 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0269 seconds
go
Elapsed time: 0.0313 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0277 seconds
go
Elapsed time: 0.0310 seconds
Elapsed time: 0.0274 seconds
go
Elapsed time: 0.0321 seconds
Elapsed time: 0.0274 seconds
go
Elapsed time: 0.0310 seconds
Elapsed time: 0.0286 seconds
go
Elapsed time: 0.0309 seconds
Elapsed time: 0.0273 seconds
go
Elapsed time: 0.0323 seconds
Elapsed time: 0.0277 seconds
go
Elapsed time: 0.0307 seconds
Elapsed time: 0.0289 seconds
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
