using JuMP, NLPModels, Optimize, LinearOperators
using LSDescentMethods, Stopping, PyPlot, LineSearch
using QRupdate

##Code de Hager-Zhang si on preconditionne en n'utilisant que les k dernieres
##directions, puisqu'on accepte la k-ieme quand on entre dans le subspace,
##meme si cette direction est consideree comme pas assez orthogonale!
##
##Version du 14 novembre 2017, qui tente de reproduire le code C.
##

include("subspace_iteration7.jl")
include("QRupdate.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)/(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function HZCG7(nlp :: AbstractNLPModel;
                    normal :: Bool=false,
                    m :: Int=11,
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
                    linesearch2 :: Function = Newarmijo_wolfe,#Linesearch du L-BFGS
                    linesearch :: Function = Newarmijo_wolfe,#linesearch du L-CG
                    #linesearch :: Function = _hagerzhang2!,#linesearch du L-CG
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

# Voici la version ultime, que l'on espère beaucoup plus rapide!
# On remplace Z' par R'*Sk. On remplace
# aussi tout les calculs de forme matrix-matrix-vector sous la forme
# maxtrix-(matrix-vector) pour sauver des opérations.


    ###point initial###
    x = copy(nlp.meta.x0)

    ###dimension inititiale###
    const n = nlp.meta.nvar
    ###Changement de la valeur de m si n est trop petit###
    if m > n
        m = n
    end

    ###Initialisation des vecteurs necessaires###
    xt = zeros(n);
    ∇ft = zeros(n);
    ∇fprec = zeros(n);
    s = zeros(x);
    y = zeros(n);
    d = zeros(n);
    subgradt = zeros(m);#sert à tester Z'∇ft
    subgrad = zeros(m);#sert à stocker Z'∇ft (pour le subspace)



    ###Initialisation des matrices necessaires###
    Sₖ = zeros(n, m)

    ###Initialisation des parametres necessaire###
    β  = 0.0
    σₖ = 1.0

    f = obj(nlp, x)
    iter = 1

    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    println("maximum(∇f) = $(maximum(∇f))")

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

        d = - ∇f + β * d
        slope = ∇f ⋅ d
    println("maximum(∇f) = $(maximum(∇f))")
        println("minimum(x) = $(minimum(x))")
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

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft;
                                                                                mayterminate = false,
                                                                                verboseLS = verboseLS,
                                                                                debug = debug,
                                                                                kwargs...)

        t *= σₖ

        if verboseLS
            verbose && print("\n")
        else
            verbose && @printf("  %4d  %8e  %8e %8e  %8e\n", nbk, t, σₖ,grad(h,t),t_original*σₖ)
        end

        ###Update of Sₖ and R###

        if iter > m #qrupdate (rank 1 update)
            R = qrupdate3!(Sₖ, R, d, iter)#TODO: allouer R alors que R change de taille???
        elseif iter == 1 #initialisation de R
            Sₖ[:,1] = d
            (Q,R) = qr(Sₖ[:,1])
        elseif iter == 2
            Sₖ[:,2] = d
            (Q,R) = qr(Sₖ[:,1:2])
        else #qraddcol (updating R while adding columns to Sₖ)
            #needs to Z-less update R, and then add "d" to Sₖ
            R      = qraddcol(Sₖ[:,1:iter-1],R,d)
            Sₖ[:,iter] = d
        end#if !normal

        iter += 1

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

        if iter > m
            subgradt = (inv(triu(R))' * (Sₖ' * ∇ft))
        else
            subgradt = (inv(R)' * (Sₖ[:,1:iter - 1]' * ∇ft))
        end

        Normee = (norm(subgradt).^2) / (norm(∇ft).^2)
        (subspace && iter != 2) && (assez_ortho = ((1-η₀) >= Normee))

        if !assez_ortho && (OK && !(optimal || tired || unbounded) )
print_with_color(:green,"subspace iteration")

            # print_with_color(:green,"subspace iteration")
            #Si la derniere direction n'est pas bonne, on la met quand meme
            #dans la matrice, on ne garde pas le nouveau point et on part
            #sur le subspace problem avec les vieilles valeurs (celles sans "t")

            subspace_activation += 1

            if iter > m
                subgrad = (Sₖ' * ∇f)
            else
                subgrad = (Sₖ[:,1:iter - 1]' * ∇f)#used for the subspace problem
            end

            I = min(iter-1,m)

            (R,xt,ft,∇ft,∇f,yt,st,d,iter,assez_ortho) = subspace_iteration7(nlp,x,f,subgrad,R,Sₖ,iter,assez_ortho,y,s,σ,m;
                                                                            η = η,
                                                                            normal=normal,
                                                                            verbose=verbose,
                                                                            verboseLS=verboseLS,
                                                                            linesearch=linesearch2,
                                                                            kwargs...)
            ∇fNorm = norm(∇ft, Inf)

            if iter > m
                subgradt = (inv(triu(R))' * (Sₖ' * ∇ft))
            else
                subgradt = (inv(R)' * (Sₖ[:,1:iter - 1]' * ∇ft))
            end


            if n <= 2
                verbose && @printf("%4d  %8e  %7.1e %24s", iter, ft, ∇fNorm, x)
            else
                verbose && @printf("%4d  %8e  %7.1e", iter, ft, ∇fNorm)
            end

            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir

        end#if du subspace_iteration

        β = 0.0

        ### Powell restart ###
        if (((∇ft⋅∇f) < 0.2 * (∇ft⋅∇ft)) || (iter > 5))
            dy = d ⋅ yt

            βₖ = ( (yt ⋅ ∇ft) - Θₖ * ( (yt ⋅ yt) * (d ⋅ ∇ft)) / dy ) / dy
            ηₖ = η * ( (d ⋅ ∇f) / (d' * d) )

            β  = max(βₖ, ηₖ)
        end

        ###update of variables###

        x = copy(xt)
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
        s = st
        y = yt
        subgrad = copy(subgradt)

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
