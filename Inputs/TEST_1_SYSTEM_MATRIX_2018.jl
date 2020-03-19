include("Build_system_matrix.jl")
#Test Build_system_matrix 3D

#In order to simplify the computation by hand of the results, we had to change
#    for z = (-_box_length_z + z_pixel_length) / 2:_dist_z:(_box_length_z - z_pixel_length) / 2
# and replace it by:
# for z = (0.0):_dist_z:(_box_length_z - z_pixel_length) / 2
#
######Input:#####
    # Build_system_matrix(2.0,2.0,2.0,2,2,2,3,3,1,1.0,2.0,deg2rad(60),deg2rad(60),0)

######Output:#####
    # 27×8 SparseMatrixCSC{Float64,Int64} with 10 stored entries:
    #   [7 ,  1]  =  0.666667
    #   [8 ,  3]  =  0.845299
    #   [9 ,  3]  =  0.666667
    #   [1 ,  5]  =  0.845299
    #   [4 ,  5]  =  0.666667
    #   [2 ,  7]  =  1.0
    #   [3 ,  7]  =  0.845299
    #   [5 ,  7]  =  0.845299
    #   [6 ,  7]  =  0.666667
    #   [2 ,  8]  =  1.0

#in order to test Gen_position, the following comments are to be uncommented:
        # println()
        # print(S_)
        # println()
        # println(P_[1,:])
        # println(P_[2,:])
        # println(P_[3,:])

#and the results should be:
# [-2.0, 0.0, 0.0]
# [1.4641, 2.0, 1.4641, 1.4641, 2.0, 1.4641, 1.4641, 2.0, 1.4641]
# [-2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0]
# [0.0, 0.0, 0.0, 2.3094, 2.3094, 2.3094, -2.3094, -2.3094, -2.3094]

#in order to have automatic testing, here is the output of the code verified with
#the 2 precedent test:
#
######Input:#####
A = Build_system_matrix(2.0,2.0,1.0,2,2,1,3,3,1,1.0,2.0,deg2rad(60),deg2rad(60),0)
######Output:#####

TSM = [0.845299 0.0      0.0      0.0      0.0      0.0      0.0      0.0;#1
       0.0      0.0      1.0      1.0      0.0      0.0      0.0      0.0;#2
       0.0      0.0      0.845299 0.0      0.0      0.0      0.0      0.0;#3
       0.0      0.0      0.0      0.0      0.976068 0.0      0.0      0.0;#4
       0.0      0.0      0.0      0.0      0.0      0.0      1.1547   0.690599;#5
       0.0      0.0      0.0      0.0      0.0      0.0      0.976068 0.0;#6
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#7
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#8
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#9
       0.0      0.0      0.0      0.0      0.845299 0.0      0.0      0.0;#10
       0.0      0.0      0.0      0.0      0.0      0.0      1.0      1.0;#11
       0.0      0.0      0.0      0.0      0.0      0.0      0.845299 0.0;#12
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#13
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#14
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#15
       0.976068 0.0      0.0      0.0      0.0      0.0      0.0      0.0;#16
       0.0      0.0      1.1547   0.690599 0.0      0.0      0.0      0.0;#17
       0.0      0.0      0.976068 0.0      0.0      0.0      0.0      0.0;#18
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#19
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#20
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#21
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#22
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#23
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#24
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#25
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0;#26
       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0];#27

TSM = sparse(TSM)
#norm(A-TSM)<0.0001 ## Not implemented as of February 16 2018
#norm(full(A)-full(TSM))<0.0001

# Testing is made by calculating the first values per hand, and once verified with
# the modified code, we switched back the code to the original one. That change
# was only on the z-height so that the values were easier to calculate.
#
# The order of the projections in the matrix is quite unusual. We calculate,
#
# 0) for each source-to-detecter-pannel pair, the vertical-central-detectors (VCD)
# For a vertical group of detector we start with the upper one, going down (Zaxis)
#
# 1)Then, we calculate the vertical-detectors that are closest to the VCD
# COUNTERCLOCKWISE. Those will also be made from up to down (in the Z-axis).
# We here define Clockwise by: Source would be the origin, VCD[2] the (0,0,1)
#
#
# 2)Then, we calculate the vertical-detectors that are closest to the VCD
# CLOCKWISE. Those will also be made from up to down (in the Z-axis).
#
# We then continue step 1) and 2) with the second closest one and etc.
#
# When we are done with a projection, we rotate the gantry (for the SSCT version)
# and we restart step 0), 1) and 2). Rotation of the gantry implies a new
# source-to-detecter-pannel pair.
#
# Once the gantry has been fully rotated, we increase the z-height
