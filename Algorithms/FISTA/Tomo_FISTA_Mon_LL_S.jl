#Application de l'algorithme FISTA/FGP, adapté au problème de reconstruction
#tomographique. On utilise donc un Log Likelihood et une Variation Totale.
#
#Le NLP voulu est un TVLOGTESTIN(pour reconstruction), il construit le TVP, le LLS
#contient entre autres les paramètres pour ne plus dépendre des pD et pR.
#
#L est la constante de Lipschitz de A. Elle vaut 0.606 pour notre matrice.
#On rajoutera sous peu un outil pour approximer cette valeur. ligne:34-36
#
#
#

# using ToeplitzMatrices
using PyPlot

include("FGP.jl")

function Tomo_FISTA_Mon_LL_S(nlp;
                    L :: Real = 0.0,
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    mon :: Bool=true,
                    epsilon :: Float64=1e-4,
                    denoiseiter :: Int=10,
                    tv :: String="iso",
                    BC :: String="reflexive",#pas encore utile,
                    kwargs...)
    if L < 0
        error("L must be positive")
    end

    n,m    = size(nlp.A);
if n > 20000
    if abs(L) < eps() #500k
        L = maximum(nlp.A'*(nlp.A * ones(m)))/18.5
        println(36)
    end
else

    if abs(L) < eps()
        L = maximum(nlp.A'*(nlp.A * ones(m)))/60.0
        println(42)
    end
end

if (size(nlp.A)[1]<20000) && (sum(nlp.b)< 200000)
    L = maximum(nlp.A'*(nlp.A * ones(m)))/12.0
    println(48)
elseif (size(nlp.A)[1]<80000) && (sum(nlp.b)< 200000)
    L = maximum(nlp.A'*(nlp.A * ones(m)))/2.0
    println(51)
end

if abs(sum(nlp.b)-121028) < 100
    L = maximum(nlp.A'*(nlp.A * ones(m)))/16.0
    println(56)
end

    fun_all = [];
    ng_all  = [];
    x_iter = copy(nlp.meta.x0)

    x_old  = zeros(m);
    y      = x_iter;
    t_new  = 1;

    P1 = [];
    P2 = [];

    # L,l,u = 0.606, 0.0, Inf
    l,u = eps(), Inf
    lambda,epsil = nlp.lambda, nlp.epsi

    grad  = zeros(m);
    vg    = zeros(m);
    N, M  = Int64(sqrt(size(nlp.A,2))),Int64(sqrt(size(nlp.A,2)))
    iter  = 1

    stp, vg = start!(nlp,stp,x_iter)
    ∇fNorm = BLAS.nrm2(m, vg, 1)
    vf = obj(nlp, x_iter)

    if n <= 2
        verbose && @printf("%4s  %8s  %11s %17s   \n", "iter", "f", "‖∇f‖", "x")
        verbose && @printf("%4d  %8e  %7.1e %24s", iter, vf, ∇fNorm,x_iter)
    else
        verbose && @printf("%4s  %8s  %11s   \n", "iter", "f", "‖∇f‖")
        verbose && @printf("%4d  %8e  %7.1e", iter, vf, ∇fNorm)
    end
    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x_iter,vf,vg)

    while (!(optimal || tired || unbounded) )

        x_old  = x_iter;
        t_old  = t_new;

        #grad   = LSQgrad!(nlp.LSQ, y, grad)        #grad = pD.A' * (pD.A*y - pD.b);#TODO grad(LSQ,y)
        grad   = grad!(nlp.LLs, y, grad)

        y      = y - (1 / L) * grad;
        Y      = reshape(y, N, M)        #Y = reshape(y, pR.n, pR.m)

        (Z_iter, it, fun_denoise, P1, P2) = fgp(Y, lambda/L,  P1, P2;
                                        l=l, u=u, kwargs...)#epsilon,tv,MAXITER,

        z_iter = Z_iter[:];

        vf,vg = objgrad!(nlp, z_iter, vg) #vf,vg,vfR,vfD,vgR,vgD = eval_costpDpR(z_iter,pD,pR;kwargs...)#TODO objgrad(LASSO,z_iter)
        vf = obj(nlp,z_iter)#patch pour monotonie GUS 20/08/2018
        ###Checks for monotony###
        if mon == 0
            x_iter = z_iter;
        else
            if iter > 1
                fun_val_old = fun_all[end];
                if vf > fun_val_old
                    x_iter = x_old;
                    vf = fun_val_old;
                else
                    x_iter = z_iter;
                end #if
            end #if
        end #if

        t_new = (1 + sqrt(1 + 4 * t_old^2)) / 2;
        y = x_iter + (t_old / t_new) * (z_iter - x_iter) + ((t_old - 1) / t_new) * (x_iter - x_old);

        iter = iter + 1

        ###Creates the obj and grad graphics###
        push!(fun_all, vf);
        push!(ng_all, norm(vg));

        if n <= 2
            verbose && @printf("%4d  %8e  %7.1e %24s", iter, vf, norm(vg), x_iter)
        else
            verbose && @printf("%4d  %8e  %7.1e", iter, vf, norm(vg))
        end

        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x_iter,vf,vg)

    end #while

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    else status = :UserLimit
    end

    h_f = nlp.counters.neval_obj
    h_g = nlp.counters.neval_grad
    h_h = nlp.counters.neval_hprod


    return (x_iter, vf, stp.optimality_residual(vg), iter, optimal, tired, status, h_f, h_g, h_h)
    #return x_iter, fun_all, ng_all
end #function
