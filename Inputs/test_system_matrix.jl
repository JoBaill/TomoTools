_box_length_x=15
_box_length_y=15
_box_length_z=15

_nb_vox_x=15
_nb_vox_y=15
_nb_vox_z=15

_nb_det_lat=10
_Dv=3

_nb_proj=10

_dist_z=1
_radius=20

_phi= pi/12
_theta= pi/4
_angle_tot= pi

include("Build_system_matrix.jl")

@time Build_system_matrix(_box_length_x,_box_length_y,_box_length_z,
                                      _nb_vox_x,_nb_vox_y,_nb_vox_z,
                                      _nb_det_lat,_Dv,_nb_proj,_dist_z,
                                      _radius,_phi,_theta,_angle_tot)
