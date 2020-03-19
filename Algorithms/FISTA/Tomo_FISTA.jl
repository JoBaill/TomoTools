#Application de l'algorithme FISTA/FGP, adapté au problème de reconstruction
#tomographique. On utilise donc un Least Square et une Variation Totale.
#
#Le NLP voulu est un TVRecon(pour reconstruction), il construit le TVP, le LSQ
#contient entre autres les paramètres pour ne plus dépendre des pD et pR.
#
#L est la constante de Lipschitz de A. Elle vaut 0.606 pour notre matrice.
#On rajoutera sous peu un outil pour approximer cette valeur.
#
#
#

# using ToeplitzMatrices
using PyPlot

include("FGP.jl")

function Tomo_FISTA(nlp;
                    Affichage :: Bool=false,
                    mon :: Bool=false,
                    epsilon :: Float64=1e-4,
                    MAXITER :: Int=100,
                    denoiseiter :: Int=10,
                    tv :: String="iso",
                    BC :: String="reflexive",#pas encore utile,
                    kwargs...)

    fun_all = [];
    ng_all  = [];

    n,m    = size(nlp.LSQ.A);
    x_iter = zeros(m);
    y      = x_iter;
    t_new  = 1;

    P1 = [];
    P2 = [];

    L,l,u = 0.606, 0.0, Inf
    lambda,epsil = nlp.lambda, nlp.epsi

    x_old = zeros(m);
    grad  = zeros(m);
    vg    = zeros(m);
    N, M   = Int64(sqrt(size(nlp.LSQ.A,2))),Int64(sqrt(size(nlp.LSQ.A,2)))

    for i = 0:MAXITER

        x_old  = x_iter;
        t_old  = t_new;

        grad   = LSQgrad!(nlp.LSQ, y, grad)        #grad = pD.A' * (pD.A*y - pD.b);#TODO grad(LSQ,y)

        y      = y - (1 / L) * grad;
        Y      = reshape(y, N, M)        #Y = reshape(y, pR.n, pR.m)

        (Z_iter, iter, fun_denoise, P1, P2) = fgp(Y, lambda/L,  P1, P2;
                                        l=l, u=u, kwargs...)#epsilon,tv,MAXITER,

        z_iter = Z_iter[:];

        vf,vg = objgrad!(nlp, z_iter, vg) #vf,vg,vfR,vfD,vgR,vgD = eval_costpDpR(z_iter,pD,pR;kwargs...)#TODO objgrad(LASSO,z_iter)

        ###Checks for monotony###
        if mon == 0
            x_iter = z_iter;
        else
            if i > 1
                fun_val_old = fun_all[end];
                if vf > fun_val_old
                    x_iter = x_old;
                    vf = fun_val_old;
                else
                    x_iter = z_iter;
                end #if
            end #if
        end #if

        ###Creates the obj and grad graphics###
        push!(fun_all, vf);
        push!(ng_all, norm(vg));

        t_new = (1 + sqrt(1 + 4 * t_old^2)) / 2;
        y = x_iter + (t_old / t_new) * (z_iter - x_iter) + ((t_old - 1) / t_new) * (x_iter - x_old);

        i = i + 1

    end #for
    if Affichage
        X = reshape(x_iter, 128, 128)
        figure(1)
        imshow(X,cmap = (ColorMap("gray")))
    end


    return x_iter, fun_all, ng_all
end #function
