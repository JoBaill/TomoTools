export validate

function validate(_box_length_x::Real,
                  _box_length_y::Real,
                  _box_length_z::Real,
                  _nb_vox_x::Int64,
                  _nb_vox_y::Int64,
                  _nb_vox_z::Int64,
                  t_matrix::Array{Float64},
                  _S::Vector,
                  _P::Array{Float64},
                  _nb_det_lat::Int64,
                  _Dv::Int64)

  a = _S[1];#coordinates of the source
  b = _S[2];#coordinates of the source
  c = _S[3];#coordinates of the source

  x_pixel_length = _box_length_x / _nb_vox_x;
  y_pixel_length = _box_length_y / _nb_vox_y;
  z_pixel_length = _box_length_z / _nb_vox_z;

  t_matrix[ (t_matrix .<= 0.0) .| (t_matrix .>= 1.0) ] = NaN;
  T1=size(t_matrix,1)
  T2=size(t_matrix,2)
######Possible erreur numérique sur le calcul de t_matrix
######

  for k = 1:_nb_det_lat * _Dv
    unique_t_vector = sort(unique(t_matrix[k,:]))'
    t_matrix[k,:] = [unique_t_vector repmat([NaN],1,size(t_matrix,2) - size(unique_t_vector,2))];
  end

  nb_vox_tot = size(t_matrix, 2)

  t_1a = (1 - t_matrix) * a;
  t_1b = (1 - t_matrix) * b;
  t_1c = (1 - t_matrix) * c;

  X_123 = [repmat(_P[1,:],1,nb_vox_tot);repmat(_P[2,:],1,nb_vox_tot);
             repmat(_P[3,:],1,nb_vox_tot)];
  X_123 = X_123 .* [t_matrix; t_matrix; t_matrix];
  X_123 = X_123 + [t_1a; t_1b; t_1c;];

  #for i = 1:_nb_det_lat * _Dv
  #      X_1[i,:] = X_123[i,:];
  #      X_2[i,:] = X_123[i+_nb_det_lat * _Dv,:];
  #      X_3[i,:] = X_123[i+(2 * _nb_det_lat * _Dv),:];
  #end
  i = _nb_det_lat * _Dv
  X_1 = X_123[1:i,:];
  X_2 = X_123[i+1:2*i,:];
  X_3 = X_123[(2*i)+1:3*i,:];



  X_1_bool = (X_1 .>= (-(_box_length_x / 2.0)-x_pixel_length*0.0000001)) .& (X_1 .<= ((_box_length_x / 2.0)+x_pixel_length*0.0000001));
  X_2_bool = (X_2 .>= (-(_box_length_y / 2.0)-y_pixel_length*0.0000001)) .& (X_2 .<= ((_box_length_y / 2.0)+y_pixel_length*0.0000001));
  X_3_bool = (X_3 .>= (-(_box_length_z / 2.0)-z_pixel_length*0.0000001)) .& (X_3 .<= ((_box_length_z / 2.0)+z_pixel_length*0.0000001));

  X_123_bool = X_1_bool .* X_2_bool .* X_3_bool;
  X_123_bool = Float64.(X_123_bool)
  X_123_bool[(X_123_bool .== 0.0)]=NaN;

  t_matrix = t_matrix .* X_123_bool;
  X_1 = X_1 .* X_123_bool;
  X_2 = X_2 .* X_123_bool;
  X_3 = X_3 .* X_123_bool;

  for i = 1: (_nb_det_lat * _Dv)
    k = sortperm(t_matrix[i,:]);
    t_matrix[i,:] = permute!(t_matrix[i,:],k)
    X_1[i,:] = permute!(X_1[i,:],k)
    X_2[i,:] = permute!(X_2[i,:],k)
    X_3[i,:] = permute!(X_3[i,:],k)
  end

  t_matrix[isnan.(t_matrix)] = 0.0;
  X_1[isnan.(X_1)] = 0.0;
  X_2[isnan.(X_2)] = 0.0;
  X_3[isnan.(X_3)] = 0.0;

(t_matrix,X_1,X_2,X_3)
end
