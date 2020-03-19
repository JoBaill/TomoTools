pi = %pi
_box_length_x = 40
_box_length_y = 40
_box_length_z = 40
 _nb_vox_x    = 30
 _nb_vox_y    = 30
 _nb_vox_z    = 30
 _radius      = 29
_nb_det_lat   = 10
_angle_tot    = pi
_dist_z       = 1
_theta        = pi/4
_nb_proj      = 9
_phi          = pi/12
_Dv           = 7

(A)=Build_system_matrix(_box_length_x,_box_length_y,_box_length_z,
                               _nb_vox_x,_nb_vox_y,_nb_vox_z,
                               _nb_det_lat,_Dv,_nb_proj,_dist_z,
                               _radius,_phi,_theta,_angle_tot);
