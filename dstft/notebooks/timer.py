import numpy as np
import re
import itertools
import torch

# Raw data including "go" lines
raw_data = """
go
Elapsed time: 0.0354 seconds
Elapsed time: 0.0277 seconds
go
Elapsed time: 0.0336 seconds
Elapsed time: 0.0279 seconds
go
Elapsed time: 0.0429 seconds
Elapsed time: 0.0286 seconds
go
Elapsed time: 0.0324 seconds
Elapsed time: 0.0274 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0262 seconds
go
Elapsed time: 0.0357 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0307 seconds
Elapsed time: 0.0276 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0309 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0301 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0305 seconds
Elapsed time: 0.0259 seconds
go
Elapsed time: 0.0308 seconds
Elapsed time: 0.0267 seconds
go
Elapsed time: 0.0301 seconds
Elapsed time: 0.0267 seconds
go
Elapsed time: 0.0314 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0313 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0305 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0318 seconds
Elapsed time: 0.0278 seconds
go
Elapsed time: 0.0321 seconds
Elapsed time: 0.0272 seconds
go
Elapsed time: 0.0321 seconds
Elapsed time: 0.0282 seconds
go
Elapsed time: 0.0323 seconds
Elapsed time: 0.0271 seconds
go
Elapsed time: 0.0317 seconds
Elapsed time: 0.0272 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0305 seconds
Elapsed time: 0.0263 seconds
go
Elapsed time: 0.0321 seconds
Elapsed time: 0.0275 seconds
go
Elapsed time: 0.0308 seconds
Elapsed time: 0.0267 seconds
go
Elapsed time: 0.0304 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0305 seconds
Elapsed time: 0.0276 seconds
go
Elapsed time: 0.0319 seconds
Elapsed time: 0.0280 seconds
go
Elapsed time: 0.0316 seconds
Elapsed time: 0.0275 seconds
go
Elapsed time: 0.0316 seconds
Elapsed time: 0.0272 seconds
go
Elapsed time: 0.0309 seconds
Elapsed time: 0.0268 seconds
go
Elapsed time: 0.0307 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0307 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0300 seconds
Elapsed time: 0.0267 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0259 seconds
go
Elapsed time: 0.0304 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0299 seconds
Elapsed time: 0.0262 seconds
go
Elapsed time: 0.0304 seconds
Elapsed time: 0.0262 seconds
go
Elapsed time: 0.0301 seconds
Elapsed time: 0.0263 seconds
go
Elapsed time: 0.0304 seconds
Elapsed time: 0.0266 seconds
go
Elapsed time: 0.0300 seconds
Elapsed time: 0.0262 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0259 seconds
go
Elapsed time: 0.0304 seconds
Elapsed time: 0.0269 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0264 seconds
go
Elapsed time: 0.0301 seconds
Elapsed time: 0.0261 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0260 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0260 seconds
go
Elapsed time: 0.0303 seconds
Elapsed time: 0.0260 seconds
go
Elapsed time: 0.0302 seconds
Elapsed time: 0.0262 seconds
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
