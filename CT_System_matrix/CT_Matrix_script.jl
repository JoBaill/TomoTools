#####Create realistic CT system matrix#####
##This script shall make a CT system matrix##
##that have realistic dimensions. Taking##
##unrealistic dimensions tend to produce##
##errors when calculating the noisy sinograms##
using JLD
include("../../SystemMatrix/Build_system_matrix.jl")

Apetit = Build_system_matrix(30.0,30.0,1.0,#dim
                        128,128,1,#voxel
                        403,1,150,#det_lat,Dv,proj
                        1.0,24.0,#dist_z radius
                        deg2rad(80),deg2rad(80)
                        ,359)

Agrand = Build_system_matrix(30.0,30.0,1.0,#dim
                        128,128,1,#voxel
                        403,1,400,#det_lat,Dv,proj
                        1.0,24.0,
                        deg2rad(80),deg2rad(80)
                        ,359)

Atresgrand = Build_system_matrix(30.0,30.0,1.0,#dim
                        256,256,1,#voxel
                        671,1,1160,#det_lat,Dv,proj
                        1.0,24.0,
                        deg2rad(80),deg2rad(80)
                        ,359)

save("Apetit.jld","Apetit",Apetit)
save("Agrand.jld","Agrand",Agrand)
save("Atresgrand.jld","Atresgrand",Atresgrand)
