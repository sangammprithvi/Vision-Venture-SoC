import numpy as np

num_students = 4
num_assignments = 5

num_grades = 5

#(a) creating the 2D array:
myarr = np.arange(start=0, stop=num_students*num_grades)

myarr = np.reshape(myarr, (num_students, num_grades))

#(b) Julie's Grades:
julie_grades = myarr[2,:]
print(julie_grades)

#(c) assignment 4 grades:
assignment_4_grades = myarr[:,4]
print(assignment_4_grades)
