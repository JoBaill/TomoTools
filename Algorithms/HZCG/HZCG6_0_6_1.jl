using JuMP, NLPModels, Optimize, LinearOperators
using LSDescentMethods, Stopping, PyPlot, LineSearch

##Code de Hager-Zhang si on preconditionne en n'utilisant que les k dernieres
##directions, puisqu'on accepte la k-ieme quand on entre dans le subspace,
##meme si cette direction est consideree comme pas assez orthogonale!
##
##Version du 14 novembre 2017, qui tente de reproduire le code C.
##

include("subspace_iteration6_1.jl")
include("QRupdate.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)/(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function HZCG6(nlp :: AbstractNLPModel;
                    normal :: Bool=false,
                    m :: Int=11,
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
                    linesearch2 :: Function = Newarmijo_wolfe,#Linesearch du L-BFGS
                    #linesearch :: Function = Newarmijo_wolfe,#linesearch du L-CG
                    linesearch :: Function = _hagerzhang2!,#linesearch du L-CG
                    scaling :: Bool = true,
                    print_h :: Bool = false,
                    print_h_iter :: Int64 = 1,
                    subspace :: Bool = true,
                    η₀ :: Float64=0.01,#Selon le code en C
                    η  :: Float64=0.4, #empiriquement, best is between (0,1)
                    η₁ :: Float64=0.9,#Selon le code en C
                    Θₖ :: Float64=1.0, #empiriquement, best is between [1,2)
                    assez_ortho :: Bool = true,
                    show_sub_activation :: Bool = false,#will print the number of times we entered the subspace
                    kwargs...)

    ###point initial###
    x = copy(nlp.meta.x0)

    ###dimension inititiale###
    n = nlp.meta.nvar

    ###Changement de la valeur de m si n est trop petit###
    if m > n
        m=n
    end

    ###Initialisation des vecteurs necessaires###
    xt = zeros(n);
    ∇ft = zeros(n);
    ∇fprec = zeros(n);
    s = zeros(x);
    y = zeros(n);
    d = zeros(n);

    ###Initialisation des matrices necessaires###
    Sₖ = zeros(n, m)
    Z  = zeros(n, m)
    R  = zeros(m, m)

    ###Initialisation des parametres necessaire###
    β  = 0.0
    σₖ = 1.0

    f = obj(nlp, x)

    iter = 1

    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    subspace_activation = 0

    if n <= 2
        verbose && @printf("%4s  %8s  %11s %17s  %13s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "x", "∇f'd", "bk","t","scale","h'(t)","t_original")
        verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm,x)
    else
        verbose && @printf("%4s  %8s  %11s %8s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "∇f'd", "bk","t","scale","h'(t)","t_original")
        verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
    end

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    while (OK && !(optimal || tired || unbounded) ) #Boucle principale du gradient conjugue
#print_with_color(:green,"normal iteration")
        d = - ∇f + β * d
#println("d = $d")
        slope = ∇f ⋅ d
#println("slope = $slope")
        verbose && @printf(" %10.1e", slope*σₖ)

        if iter == 1
            h = LineModel(nlp, x, d * σₖ)
        else
            h = Optimize.redirect!(h, x, d * σₖ)
        end#if iter

        debug = false

        if print_h && (iter == print_h_iter)
            debug= true
            graph_linefunc(h, f, slope*σₖ;kwargs...)
        end

        verboseLS && println(" ")

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft;mayterminate = false, verboseLS = verboseLS, debug = debug, kwargs...)

        t *= σₖ

        if verboseLS
            verbose && print("\n")
        else
            verbose && @printf("  %4d  %8e  %8e %8e  %8e\n", nbk, t, σₖ,grad(h,t),t_original*σₖ)
        end

        # if linesearch in interfaced_algorithms
        #     ft = obj(nlp, x + t * d)
        #     nlp.counters.neval_obj += -1
        # end#if in interfaced_algorithms

        ###Update of Sₖ###
        #normal = true ###TODO:enlever cette ligne quand le fancy QR fonctionnera###

        if !normal #Si on veut utiliser QR fancy, pas encore completement fonctionnel(slow).
            (Z,R) = MyZupdate(Sₖ,copy(Z),copy(R),iter,copy(d),m)
            iter += 1
        else
            it       = mod(iter - 1,m) + 1
            Sₖ[:,it] = d
            Z,R      = qr(Sₖ[:,1:min(iter,m)])
            iter += 1
        end#if !normal

        ###update du pas###
        xt = x + t * d

        ###calcul le gradient au nouveau point si pas deja fait###
        good_grad || (∇ft = grad!(nlp, xt, ∇ft))

        st = xt - x
        yt = ∇ft - ∇f
        β = 0.0

        ###Barzilai-Borwein scaling###
        σ = σₖ
        if scaling
            σₖ = BB(st,yt)
        end#if scaling

        ∇fNorm = norm(∇ft, Inf)
        if n <= 2
            verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm, x)
        else
            verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
        end


        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
        OK = !stalled_linesearch & !stalled_ascent_dir

        ###on verifie l'orthogonalite avec: (1-η₀²) ⋝ ||∇f(x)ᵗ * Z|| / ||∇f(x)|| (equation 3.4)###
        assez_ortho = true
        Normee = (norm((∇ft)'*Z[:,1:min(iter-1,m)]).^2) / (norm(∇ft).^2)
        (subspace && iter != 2) && (assez_ortho = ((1-η₀) >= Normee))

        if !assez_ortho && (OK && !(optimal || tired || unbounded) )
            print_with_color(:green,"subspace iteration")
            #Si la derniere direction n'est pas bonne, on la met quand meme
            #dans la matrice, on ne garde pas le nouveau point et on part
            #sur le subspace problem avec les vieilles valeurs (celles sans "t")
            subspace_activation += 1
            I = min(iter-1,m)

            (Z,R,xt,ft,∇ft,∇f,yt,st,d,iter,assez_ortho) = subspace_iteration6(nlp,x,f,∇f,m,Z[:,1:I],R,Sₖ,iter,assez_ortho,y,s,σ;
                                                                                    η = η, normal=normal,verbose=verbose,
                                                                                    verboseLS=verboseLS,linesearch=linesearch2,kwargs...)

#println()
#print(R)
            ∇fNorm = norm(∇ft, Inf)
            if n <= 2
                verbose && @printf("%4d  %8e  %7.1e %24s", iter, ft, ∇fNorm, x)
            else
                verbose && @printf("%4d  %8e  %7.1e", iter, ft, ∇fNorm)
            end
            #println("")
            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir

        end#if du subspace_iteration

        # if n <= 2
        #     verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm, x)
        # else
        #     verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
        # end
        #
        # optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
        # OK = !stalled_linesearch & !stalled_ascent_dir

        β = 0.0

        ### Powell restart ###
        if (∇ft⋅∇f) < 0.2 * (∇ft⋅∇ft)
            dy = d ⋅ yt

            βₖ = ( (yt ⋅ ∇ft) - Θₖ * ( (yt ⋅ yt) * (d ⋅ ∇ft)) / dy ) / dy
            ηₖ = η * ( (d ⋅ ∇f) / (d' * d) )

            β  = max(βₖ, ηₖ)
        end
#println("β = $β")

        ###update of variables###

        x = copy(xt)
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
        s = st
        y = yt
# if iter > 3
#     error()
# end

        if scaling ###may seems repetitive
            σₖ = BB(s,y)
        end#if scaling

    end#while

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    h_f = nlp.counters.neval_obj
    h_g = nlp.counters.neval_grad
    h_h = nlp.counters.neval_hprod

    iter = iter - 1###because we started iter at 1 and not at 0###

    println("subspace_activation = $subspace_activation")

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h_f, h_g, h_h)
end#function
