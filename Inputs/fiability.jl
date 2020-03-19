## Les coordonnées par défaut sont celles correspondant à une région d'intérêt
## pour le Shepp_Logan 128x128. Pour que la métrique soit efficace, il est
## suggéré de prendre une région d'intérêt ayant des valeurs non nulles, et
## étant supposé être lisse. Comme ça, le bruit sera hautement pénalisé.
## fiability génère les données nécessaires à la production du graph montrant
## la relation ente la qualité de l'image reconstruite et les ressources
## nécessaires à l'obtention de cette qualité.
##
## Si SSIM est include, method peut valoir SSIM, sinon les choix possibles sont:
## PSNR, SNR, MSE

include("../Metrics.jl")

function fiability( X :: AbstractArray{T}, #noisy image
                    Image :: AbstractArray{T}, #real image
                    increase :: Integer, #number of projection used for the call
                    proj_graph :: AbstractArray{Int64,1},
                    values :: AbstractArray{T,1};
                    #coordinates :: Array{UnitRange{Int64},1} = [50:80,30:65],#LS128
                    coordinates :: Array{UnitRange{Int64},1} = [45:80,35:75],#LL128
                    #coordinates :: Array{UnitRange{Int64},1} = [12:72,57:110],#Deathstar128
                    #coordinates :: Array{UnitRange{Int64},1} = [100:160,60:130],#LS256
                    #coordinates :: Array{UnitRange{Int64},1} = [90:160,70:150],#LL256
                    # coordinates :: Array{UnitRange{Int64},1} = [24:144,114:220],#Deathstar256
                    method :: Function = PSNR,
                    kwargs...) where T

    shape = Int(sqrt(length(X)));
    XX = reshape(X, shape, shape)[coordinates[1],coordinates[2]]
    I = reshape(Image, shape, shape)[coordinates[1],coordinates[2]]


    proj = proj_graph[end] + increase
    #println("proj = $proj")
    push!(proj_graph, proj)
    value = method(XX,I)
    push!(values,value)

    return proj_graph, value
end

    # if TESTING
    #     PyPlot.plot(proj_graph,PSNR_MLEM)
    # end
