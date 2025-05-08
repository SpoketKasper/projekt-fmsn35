import numpy as np
import re
import itertools
import torch

x = np.array([2.33333333e-05, 1.13333333e-04, 4.73333333e-04, 2.16666667e-03])  # 0.0006666666666666666
print(x/x.sum())
# Raw data including "go" lines
raw_data = """
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0076 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0164 seconds
go
Elapsed time: 0.0041 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0076 seconds
go
Elapsed time: 0.0036 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0075 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0075 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0132 seconds
go
Elapsed time: 0.0037 seconds
Elapsed time: 0.0010 seconds
Elapsed time: 0.0090 seconds
go
Elapsed time: 0.0040 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0077 seconds
go
Elapsed time: 0.0040 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0082 seconds
go
Elapsed time: 0.0041 seconds
Elapsed time: 0.0028 seconds
Elapsed time: 0.0080 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0076 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0111 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0080 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0091 seconds
go
Elapsed time: 0.0050 seconds
Elapsed time: 0.0010 seconds
Elapsed time: 0.0113 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0108 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0119 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0096 seconds
go
Elapsed time: 0.0050 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0117 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0116 seconds
go
Elapsed time: 0.0041 seconds
Elapsed time: 0.0010 seconds
Elapsed time: 0.0149 seconds
go
Elapsed time: 0.0041 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0085 seconds
go
Elapsed time: 0.0045 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0090 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0018 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0059 seconds
Elapsed time: 0.0015 seconds
Elapsed time: 0.0107 seconds
go
Elapsed time: 0.0044 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0091 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0040 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0086 seconds
go
Elapsed time: 0.0045 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0083 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0083 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0083 seconds
go
Elapsed time: 0.0044 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0094 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0073 seconds
go
Elapsed time: 0.0038 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0041 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0072 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0077 seconds
go
Elapsed time: 0.0037 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0083 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0039 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0078 seconds
go
Elapsed time: 0.0037 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0073 seconds
go
Elapsed time: 0.0040 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0075 seconds
go
Elapsed time: 0.0040 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0074 seconds
go
Elapsed time: 0.0037 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0075 seconds
go
Elapsed time: 0.0042 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0073 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0009 seconds
Elapsed time: 0.0072 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0075 seconds
go
Elapsed time: 0.0043 seconds
Elapsed time: 0.0008 seconds
Elapsed time: 0.0072 seconds
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
