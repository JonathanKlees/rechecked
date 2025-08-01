import subprocess
import itertools
import re

# This script runs the compare_annotations.py script with different configurations
# and collects the number of label errors for each configuration in a table format.

probability_thresholds = [0.5, 0.8]
bbox_heights = [0, 25, 40]
filter_dont_care_regions = [False, True]

results = []

pattern = re.compile(r"Overlooked:\s*(\d+)\s*,\s*Misfitting:\s*(\d+)")

for  filter_dont_care, bbox_height, prob_thresh in itertools.product(filter_dont_care_regions, bbox_heights, probability_thresholds):
    cmd = [
        "python", "compare_annotations.py",
        "--prob_thresh", str(prob_thresh),
        "--min_height", str(bbox_height),
        "--filter_dont_care", "yes" if filter_dont_care else "no"
    ]
    
    try:
        output = subprocess.check_output(cmd, text=True)
        
        match = pattern.search(output)
        if match:
            overlooked = int(match.group(1))
            misfitting = int(match.group(2))
            results.append((prob_thresh, bbox_height, filter_dont_care, overlooked, misfitting))
        else:
            print(f"No match in output: {output}")
    except Exception as e:
        print(f"Failed for params: {prob_thresh}, {bbox_height}, {filter_dont_care}")
        print(e)

# Print table

def center(text, width):
    return f"{text:^{width}}"

print("#" * 72)
print(center("Number of label errors for different configurations", 72))
print("#" * 72)

# Header Row 1
col_width = 14
print("#" + center("p > 0.5", col_width * 2 ) + "#" * 4 + center("p > 0.8", col_width * 2) + "#")

# Header Row 2
print(
    "#" + center("overlooked", col_width) + center("misfitting", col_width) + "####" +
    center("overlooked", col_width) + center("misfitting", col_width) + "#"
)

print("#" * 72)
print(center("Without don't care regions", 72))
print("#" * 72)

for i in range(len(results)):
    if i % 2 == 0:
        o1, m1 = results[i][3], results[i][4]
        o2, m2 = results[i + 1][3], results[i + 1][4]
        print(
            "#" + f"{o1:^{col_width}}" + f"{m1:^{col_width}}" + "####" +
            f"{o2:^{col_width}}" + f"{m2:^{col_width}}" + "#"
        )
    if i == 5:
        print("#" * 72)
        print(center("With don't care regions", 72))
        print("#" * 72)