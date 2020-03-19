export Make_ray

function Make_ray(_S::Vector, _P::Array{Float64},
                  _nb_det_lat::Int64, _Dv::Int64,
                  _box_length_x,
                  _box_length_y,
                  _box_length_z,
                  _nb_vox_x::Int64,
                  _nb_vox_y::Int64,
                  _nb_vox_z::Int64)


#  Def:using the (1-t)*S+t*P(:,i) parametrisation of a line, between a starting
#      and a finish point, we generate the plans that defines the limits of each
#      voxel, and find the time when our source-detector line cut them.
#      After sorting them in a matrix, we find the euclidian norm between
#      consecutive points to find the line/voxel intersection's length
#
#            _S :A column vector that contains the coordinates of the source
#            _P :A matrix of size 3 x _nb_det_lat containing the coordinates of
#                each detector's center
#   _nb_det_lat :number of lateral detector on the detector panel
# _box_length_x :x length of the bounding box defining the image
# _box_length_y :y length of the bounding box defining the image
# _box_length_z :z length of the bounding box defining the image
#     _nb_vox_x :number of voxel on the x axis(inside the box)
#     _nb_vox_y :number of voxel on the y axis(inside the box)
#     _nb_vox_z :number of voxel on the z axis(inside the box)
#

    a,b,c = _S#coordinates of the source

    x_pixel_length = _box_length_x / _nb_vox_x
    y_pixel_length = _box_length_y / _nb_vox_y
    z_pixel_length = _box_length_z / _nb_vox_z

#defines the values of x,y,z for the boundary of each voxel
    x_vector = -(_box_length_x / 2):x_pixel_length:(_box_length_x/2);
    y_vector = -(_box_length_y / 2):y_pixel_length:(_box_length_y/2);
    z_vector = -(_box_length_z / 2):z_pixel_length:(_box_length_z/2);

#replicates the matrices x-a,y-b and z-c for each values of x,y,z
    x_a_matrix = repmat(x_vector', _nb_det_lat * _Dv, 1) - a;
    y_b_matrix = repmat(y_vector', _nb_det_lat * _Dv, 1) - b;
    z_c_matrix = repmat(z_vector', _nb_det_lat * _Dv, 1) - c;

#concatenate the 3 matrices
    xyz_abc_matrix = [x_a_matrix y_b_matrix z_c_matrix];

#gives a vector containing the x value of each
#detector's center, on which we substract the x value of the source
    x_1_a_vector = _P[1, :] - a;
    x_2_b_vector = _P[2, :] - b;#idem with y
    x_3_c_vector = _P[3, :] - c;#idem with z

#gives the matrix that contains lines of x_1-a for each x_1,same for x_2,x_3
    x_1_a_matrix = repmat(x_1_a_vector, 1, length(x_vector));
    x_2_b_matrix = repmat(x_2_b_vector, 1, length(y_vector));
    x_3_c_matrix = repmat(x_3_c_vector, 1, length(z_vector));

    x1x2x3_abc_matrix = [x_1_a_matrix x_2_b_matrix x_3_c_matrix];

#fix value really close to 0 at 0
    xyz_abc_matrix[(xyz_abc_matrix .>= -9.0e-8).&(xyz_abc_matrix .<= 9.0e-8)]=0.0;
    x1x2x3_abc_matrix[(x1x2x3_abc_matrix .>= -9.0e-8).&(x1x2x3_abc_matrix
    .<= 9.0e-8)] = 0.0;

#   t=(x-a)/(d-a) with a = the x coordinate of the source, d the x coordinate
# of the actual detector. t_matrix hence contains all the t for all the
# intersection for each ray.
    #println(size(xyz_abc_matrix))
    #println(size(x1x2x3_abc_matrix))
    t_matrix = xyz_abc_matrix ./ x1x2x3_abc_matrix;


end
