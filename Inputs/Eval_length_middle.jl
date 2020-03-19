export Eval_length_middle

function  Eval_length_middle(t_matrix::AbstractArray{Float64},
                             X_1::Array{Float64},
                             X_2::Array{Float64},
                             X_3::Array{Float64},
                             _nb_det_lat::Int64,_Dv::Int64)

  nb_vox_tot = size(t_matrix, 2);

  L=[size(X_1,2) size(X_2,2) size(X_3,2)]

  diff_X1_sq = X_1[:,2:L[1]]-X_1[:,1:L[1]-1]
  #diff(X_1,2) #plus d'allocation...
  diff_X1_sq = diff_X1_sq .^2.0

  diff_X2_sq = X_2[:,2:L[2]]-X_2[:,1:L[2]-1]
  #diff(X_1,2) #plus d'allocation...
  diff_X2_sq = diff_X2_sq .^2.0

  diff_X3_sq = X_3[:,2:L[3]]-X_3[:,1:L[3]-1]
  #diff(X_1,2) #plus d'allocation...
  diff_X3_sq = diff_X3_sq .^2.0

  X_1_m = zeros(_nb_det_lat * _Dv,nb_vox_tot);
  X_2_m = zeros(_nb_det_lat * _Dv,nb_vox_tot);
  X_3_m = zeros(_nb_det_lat * _Dv,nb_vox_tot);

  Mat_norm = zeros(_nb_det_lat * _Dv,nb_vox_tot);

  for i = 1:(_nb_det_lat * _Dv)
    nb_non_zero = size(find(t_matrix[i,:]),1);

    if nb_non_zero == 0

      continue

    else
      for j = 1:(nb_non_zero - 1)
        Mat_norm[i,j] = sqrt(diff_X1_sq[i,j]+diff_X2_sq[i,j]+diff_X3_sq[i,j]);

        X_1_m[i,j] = (X_1[i,j] + X_1[i,j+1])/2.0;
        X_2_m[i,j] = (X_2[i,j] + X_2[i,j+1])/2.0;
        X_3_m[i,j] = (X_3[i,j] + X_3[i,j+1])/2.0;
      end
    end
  end

  (Mat_norm,X_1_m,X_2_m,X_3_m)
end
