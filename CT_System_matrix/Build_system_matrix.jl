export Build_system_matrix

include("Eval_length_middle.jl")
include("Gen_position.jl")
include("Make_ray.jl")
include("Rotate_gantry.jl")
include("Validate.jl")

function Build_system_matrix(_box_length_x::Real,
                             _box_length_y::Real,
                             _box_length_z::Real,
                             _nb_vox_x::Int64,
                             _nb_vox_y::Int64,
                             _nb_vox_z::Int64,
                             _nb_det_lat::Int64,_Dv::Int64,_nb_proj::Int64,
                             _dist_z,
                             _radius,_phi,_theta,_angle_tot)

    # User define a bounding box with length on the 3 usual axis of _box_length_x
    # _box_length_y and _box_length_z
    # When looking the gantry through the hole(like if the gantry was a circle),
    # the x-axis is the width of the gantry, the y-axis the height and the z-axis
    # is the one we are looking through.
    #
    # the image is then discretise into a certain number of voxel for each axis
    # (_nb_vox_x,_nb_vox_y,_nb_vox_z). The dimension of a voxel will then be
    # _box_length_x/_nb_vox_x,_box_length_y/_nb_vox_y and_box_length_z/_nb_vox_z.
    # N.B. that it's not always cubic.
    #
    # the radius of the source to origin and of the origin to detector_plane_center
    # will be equal to _radius
    #
    # _theta is the angular width of the ray and _phi is the angular thickness
    #
    # the total angle coverage for a slice in the SSCT case will be _angle_tot
    # angles are in rad
    #
    # the number of projection will be _nb_proj. The projections are made with
    # equal angles of _angle_tot/_nb_proj.
    #
    # the detector pannel has the shape of a "polygonal arc". Each of its sides
    # will be described as a lateral detector. In MSCT, the lateral dectectors are
    # subdivided in vertical detectors.
    #
    # _nb_det_lat is the number of lateral detectors. this number is really
    # important as the matrix will be made with the intersection of the rays with
    # the voxels for each source-to-detector_center pairing.
    #
    # detectors size is defined by the constant angle(theta/nb_proj).*************
    #
    # _dist_z is the distance on the z-axis between each slice.
    #
    # _Dv is the number of vertical detector
    #
    # Return the system Matrix A, containing the length of intersections for each
    # rays with each voxel of the object.

    P = zeros(3, _nb_det_lat);
    S = [0.0, 0.0, 0.0];


    epss=0.000000000000001
    ##Test on inputs
    d = sqrt(((_box_length_x / 2.0)^2.0) + ((_box_length_y / 2.0)^2.0))
    if _radius <= d + 0.01 * _radius
        println("error, _radius is too small => the source is inside the object")
        return "the algorithm has stopped before completion"
    end

    if _theta >= 2*pi
        println("error, _theta is too big => useless projections")
        return "the algorithm has stopped before completion"
    end

    x_pixel_length = _box_length_x / _nb_vox_x;
    y_pixel_length = _box_length_y / _nb_vox_y;
    z_pixel_length = _box_length_z / _nb_vox_z;

    if _box_length_z == _dist_z
        z_step = 1;
    else
        z_step = trunc(Int64,_box_length_z/_dist_z) + 1
    end

    A = sparse([1],[1],[0],_nb_det_lat * _Dv * _nb_proj * z_step,_nb_vox_x * _nb_vox_y * _nb_vox_z);#TODO initier

    i = 1:(_nb_det_lat * _Dv)

    for z = (-_box_length_z + z_pixel_length) / 2.0:_dist_z:(_box_length_z - z_pixel_length) / 2.0
#    for z = (0.0):_dist_z:(_box_length_z - z_pixel_length) / 2#TEST GUS DSA
        (S,P) = Gen_position(_radius, _theta, z, _nb_det_lat,_phi,_Dv,S , P)
        # S_=S;
        # P_=P;
        # println()
        # print(S_)
        # println()
        # println(P_[1,:])
        # println(P_[2,:])
        # println(P_[3,:])


        for k = 1:_nb_proj
            (t_matrix) = Make_ray(S, P, _nb_det_lat, _Dv, _box_length_x,
                   _box_length_y,_box_length_z, _nb_vox_x, _nb_vox_y, _nb_vox_z)

            (t_matrix,X_1,X_2,X_3)= validate(_box_length_x,_box_length_y,
            _box_length_z, _nb_vox_x, _nb_vox_y, _nb_vox_z,t_matrix, S,P,_nb_det_lat,_Dv);

            (Mat_norm,X_1_m,X_2_m,X_3_m) = Eval_length_middle(t_matrix, X_1, X_2
                                                              , X_3, _nb_det_lat,_Dv);
            # println("t_matrix=$(size(t_matrix))")
            # println("size(X_1)=$(size(X_1))")
            # println("size(X_1_m)=$(size(X_1_m))")

            a = Array{Int64,2}(round.((X_1_m+epss + (_box_length_x - x_pixel_length) / 2.0) / x_pixel_length))
            a = min.(a,_nb_vox_x-1)
            a = max.(a,0)

            b = Array{Int64,2}(round.((X_2_m+epss + (_box_length_y - y_pixel_length) / 2.0) / y_pixel_length))
            b = max.(b,0)
            b = min.(b,_nb_vox_y-1)

            c = Array{Int64,2}(round.((X_3_m+epss + (_box_length_z - z_pixel_length) / 2.0) / z_pixel_length))
            c = max.(c,0)
            c = min.(c,_nb_vox_z-1)

            l            = a + 1 + b * (_nb_vox_x) + c * (_nb_vox_x) * (_nb_vox_y);

            testo        = repmat(i', size(Mat_norm, 2), 1);

            testo2       = reshape(testo, length(testo));
            testa        = Int.(reshape(l', length(l)));

            #position     = [testo2 testa];
            #numerical trick to avoid for loops
            vec_Mat_norm = reshape(Mat_norm', length(Mat_norm));
            ggg1 = size(A,1)
            ggg2 = size(A,2)
            A            = A + sparse(testo2,testa,vec_Mat_norm,ggg1,ggg2);#, [size(A)]

            i = i + _nb_det_lat * _Dv

            (S, P) = Rotate_gantry(_angle_tot, _nb_proj, S, P, k);



        end
    end
    return A
end
