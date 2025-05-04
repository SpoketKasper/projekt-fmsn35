import numpy as np
import re
import itertools
import torch

# Raw data including "go" lines
raw_data = """
go
Elapsed time: 0.0741 seconds
go
Elapsed time: 0.0774 seconds
go
Elapsed time: 0.0675 seconds
go
Elapsed time: 0.0685 seconds
go
Elapsed time: 0.0699 seconds
go
Elapsed time: 0.0737 seconds
go
Elapsed time: 0.0751 seconds
go
Elapsed time: 0.0728 seconds
go
Elapsed time: 0.0739 seconds
go
Elapsed time: 0.0777 seconds
go
Elapsed time: 0.0722 seconds
go
Elapsed time: 0.0782 seconds
go
Elapsed time: 0.0686 seconds
go
Elapsed time: 0.0703 seconds
go
Elapsed time: 0.0684 seconds
go
Elapsed time: 0.0602 seconds
go
Elapsed time: 0.0725 seconds
go
Elapsed time: 0.0605 seconds
go
Elapsed time: 0.0588 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0594 seconds
go
Elapsed time: 0.0689 seconds
go
Elapsed time: 0.0736 seconds
go
Elapsed time: 0.0594 seconds
go
Elapsed time: 0.0595 seconds
go
Elapsed time: 0.0611 seconds
go
Elapsed time: 0.0601 seconds
go
Elapsed time: 0.0704 seconds
go
Elapsed time: 0.0596 seconds
go
Elapsed time: 0.0608 seconds
go
Elapsed time: 0.0697 seconds
go
Elapsed time: 0.0717 seconds
go
Elapsed time: 0.0694 seconds
go
Elapsed time: 0.0674 seconds
go
Elapsed time: 0.0585 seconds
go
Elapsed time: 0.0669 seconds
go
Elapsed time: 0.0673 seconds
go
Elapsed time: 0.0704 seconds
go
Elapsed time: 0.0654 seconds
go
Elapsed time: 0.0593 seconds
go
Elapsed time: 0.0597 seconds
go
Elapsed time: 0.0596 seconds
go
Elapsed time: 0.0591 seconds
go
Elapsed time: 0.0598 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0614 seconds
go
Elapsed time: 0.0597 seconds
go
Elapsed time: 0.0606 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0594 seconds
go
Elapsed time: 0.0591 seconds
go
Elapsed time: 0.0589 seconds
go
Elapsed time: 0.0585 seconds
go
Elapsed time: 0.0598 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0613 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0593 seconds
go
Elapsed time: 0.0601 seconds
go
Elapsed time: 0.0608 seconds
go
Elapsed time: 0.0613 seconds
go
Elapsed time: 0.0590 seconds
go
Elapsed time: 0.0595 seconds
go
Elapsed time: 0.0590 seconds
go
Elapsed time: 0.0716 seconds
go
Elapsed time: 0.0707 seconds
go
Elapsed time: 0.0584 seconds
go
Elapsed time: 0.0593 seconds
go
Elapsed time: 0.0588 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0596 seconds
go
Elapsed time: 0.0616 seconds
go
Elapsed time: 0.0593 seconds
go
Elapsed time: 0.0604 seconds
go
Elapsed time: 0.0586 seconds
go
Elapsed time: 0.0613 seconds
go
Elapsed time: 0.0585 seconds
go
Elapsed time: 0.0604 seconds
go
Elapsed time: 0.0594 seconds
go
Elapsed time: 0.0599 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0598 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0617 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0737 seconds
go
Elapsed time: 0.0586 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0588 seconds
go
Elapsed time: 0.0611 seconds
go
Elapsed time: 0.0660 seconds
go
Elapsed time: 0.0591 seconds
go
Elapsed time: 0.0595 seconds
go
Elapsed time: 0.0598 seconds
go
Elapsed time: 0.0595 seconds
go
Elapsed time: 0.0592 seconds
go
Elapsed time: 0.0594 seconds
go
Elapsed time: 0.0620 seconds
go
Elapsed time: 0.0602 seconds
go
Elapsed time: 0.0600 seconds
go
Elapsed time: 0.0593 seconds
go
Elapsed time: 0.0603 seconds
go
Elapsed time: 0.0598 seconds
go
Elapsed time: 0.0607 seconds
go
Elapsed time: 0.0595 seconds
go
Elapsed time: 0.0611 seconds
go
Elapsed time: 0.0589 seconds
go
Elapsed time: 0.0597 seconds
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
