export Rotate_gantry

function Rotate_gantry(_angle_tot::Any, _nb_proj::Int64, _S::Vector, _P::Array{Float64,} ,ww::Int64)


#_angle_tot=pi
#_nb_proj=3
#ww=3
#  Def: Rotate the grantry, source and detector panel, around the origin by an
# angle of _angle_tot/_nb_proj around the z-axis
#  _angle_tot : Total angle coverage per acquistion
#  _nb_proj : Number of projection done per acquisition
#  _S : A column vector that contains the coordinates of the source
#  _P : A matrix of size 3 x _nb_det_lat containing the coordinates of each
#  detector's center
#  Return :
# 	  rotated_S : new position of the source
# 	  rotated_P : new position of the panel
#
#      w is the grantry rotation angle, in the dsa axis, done per acquisition

if _nb_proj == 1
    w = 0
else
    w = _angle_tot / (_nb_proj - 1);
end

grantry_rotation_matrix = [cos(w * ww) -sin(w * ww)    0.0;
                           sin(w * ww) cos(w * ww)     0.0;
                                0.0       0.0          1.0]

rotated_P = grantry_rotation_matrix * _P;#rotate the detector panel
rotated_S = grantry_rotation_matrix * _S;#rotate the source.
#rotated_S = rotated_S'

#fix value really close to 0 at 0
    rotated_P[(rotated_P .<= 9.0e-11).&(rotated_P .>= -9.0e-11)] = 0.0;
    rotated_S[(rotated_S .<= 9.0e-11).&(rotated_S .>= -9.0e-11)] = 0.0;
(rotated_S,rotated_P)
end
