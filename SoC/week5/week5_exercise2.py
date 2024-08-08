import numpy as np

num_keypoints = 7
num_joints = 5


# All Z's in one plane, but makes it easier to see XYZ vs Start/end
keypoint_positions = np.array(
    [
        [0, 1, 0], #Head
        [0, 0, 0], #Torso
        [1, 0, 0], #Right Arm
        [-1, 0, 0], #Left Arm
        [0, -1, 0], #Lower ,Torso
        [1, -2, 0], #Right Leg
        [-1, -2, 0] #Left Leg
    ]
)

#   O
#  _|_
#   |
#  /\
joints = np.array([
    # Head to torso
    [0, 1],
    # Torso to Right arm
    [1, 2],
    # Torso to Left Arm
    [1, 3],
    # Torso to Lower Torso
    [3, 4],
    # Lower Torso to Right Leg
    [4, 5],
    # Lower Torso to Left Leg
    [4, 6]
])

# (a)Create a matrix of joint starts, and another matrix of joint ends, each of shape (num_joints, 3). 
#    The starts table should contain the position of the start of each joint (according to position)

# Write your solution below.

#(a) Create a matrix of joint starts, and another matrix of joint ends, each of shape (num_joints, 3). 
#    The starts table should contain the position of the start of each joint (according to position)

joint_starts = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [-1, 0, 0],
                         [0, -1, 0],
                         [0, -1, 0]])
print(f"Joint start points:\n {joint_starts}")

joint_ends = np.array([[0, 0, 0], #Torso
                      [1, 0, 0], #Right Arm
                      [-1, 0, 0], #Left Arm
                      [0, -1, 0], #Lower ,Torso
                      [1, -2, 0], #Right Leg
                      [-1, -2, 0]]) #Left Leg
print(f"Joint end points:\n {joint_ends}")

#(b) Create a matrix of joint-displacements, of shape (num_joints, 3). 
#    Each row represents a joint. The columns should be the difference in X, Y, and Z 
#    between the start of the joint, and the end of the joint, 
#    respectively (endX - startX, endY - startY, endZ-startZ).

joint_displacements = joint_ends - joint_starts
print(f"Joint displacements:\n {joint_displacements}")

#(c) Find the magnitude (length) of each of these displacement vectors, 
#    and output the results in an array of length num_joints. Remember the power operator is **.

joint_displacement_lengths = np.zeros(6)

for i in range(6):
    l1 = joint_displacements[i,0]**2
    l2 = joint_displacements[i,1]**2
    l3 = joint_displacements[i,2]**2

    length = (l1 + l2 + l3)**(0.5)
    joint_displacement_lengths[i] = length

print(f"Vector lengths:\n {joint_displacement_lengths}")