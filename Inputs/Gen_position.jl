export Gen_position

function Gen_position(_radius, _theta, _z, _nb_det_lat::Int64,_phi,_Dv::Int64,S,P)#(S,P)=Gen_position()

#test
#Gen_position(20, pi, 0, 11,pi/12,3)
#P =
#[-20.0   -7.63932    3.51141   12.3607  …   3.51141  -7.63932  -20.0
#-40.0  -38.0423   -32.3607   -23.5114     32.3607   38.0423    40.0
# 0.0    0.0        0.0        0.0        -5.2661   -5.2661    -5.2661]
#
#S = [-20 0 0]
#
#

 #  Def: Gen_position is a function that generates the coordinates of the source
 #         and of the detectors.
 # _radius : spacing between the source and the origin, which is the same as the
 #          spacing between the source and the detector panel center
 # _theta : The angular width of the ray
 # _z : actual z position of the slice for the current acquisition
 # _nb_det_lat : number of lateral detector on the detector panel
 # Return :
 #    S:column vector containing the coordinates of the source
 #    P:3 x _nb_det_lat matrix containing the coordinates of each detector's center
 #
 #
 # Making the central_detector using the 2*source_origin_vector actually takes
 # the origin of the actual slice (0,0,k). The vector is then multiplied per 2
 #
 # so we have the detector panel at a distance of 2*r
 #
 # then the vector is rotated around the z axis of theta/(nb_det_lat-1) degrees
 #  u times, on each side
 # where u=(nb_det_lat-1)/2
  if mod(_Dv,2) == 0
    println("error, odd number of vertical detector wanted")
    return
  end

  if _nb_det_lat == 1
        w = 0.0
  else
        w = _theta / (_nb_det_lat - 1);
  end

  P = zeros(3, _nb_det_lat);
  S = [-_radius, 0.0, _z];
  central_detector_pos = [_radius 0.0 _z];
  w = _theta / (_nb_det_lat - 1);
  println(_nb_det_lat)
  println((_nb_det_lat - 1)/ 2.0)
  g = Int64((_nb_det_lat - 1)/ 2.0) ;
  j = 1;
  for i = -g:g
    R = [cos(i * w) -sin(i * w) 0.0   S[1];#rotation matrix
        sin(i * w)  cos(i * w)  0.0   S[2];
        0.0         0.0         1.0   0.0;
        0.0         0.0         0.0   1.0;]
    T = [1.0 0.0 0.0 -S[1]; #translation matrix
         0.0 1.0 0.0 -S[2];
         0.0 0.0 1.0  0.0;
         0.0 0.0 0.0  1.0;]
    P[:,j] = (R * T * [central_detector_pos 1.0]')[1:3];
    j = j + 1;
  end
  P_ = copy(P);
  P = [P zeros(3,_nb_det_lat * (_Dv - 1))]
  if _Dv == 1
    P[(P .<= 9.e-11).&(P .>= -9.e-11)] = 0.0;
    S[(S .<= 9.e-11).&(S .>= -9.e-11)] = 0.0;
    return(S,P)
  else
    z = (_radius * tan(_phi / 2.0) * 4.0) / (_Dv - 1)#height of vert_detectors
    for k = 1:_Dv
      P[:,(((k-1) * _nb_det_lat + 1):(k * _nb_det_lat))] = P_ +
      repmat((-1)^k * ceil((k-1)/2) * z * [0;0;1],1,_nb_det_lat)
    end
    P[(P .<= 9.e-11).&(P .>= -9.e-11)] = 0.0;
    S[(S .<= 9.e-11).&(S .>= -9.e-11)] = 0.0;
  end
  (S,P)
end
