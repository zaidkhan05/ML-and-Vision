import math


# Given values
b = 127
c = 164
angle_B = 47  # degrees

# Convert angle B to radians for calculation
angle_B_rad = math.radians(angle_B)

# Step 1: Law of Cosines to find side a
a_squared = b**2 + c**2 - 2 * b * c * math.cos(angle_B_rad)
a = math.sqrt(a_squared)
print(a)