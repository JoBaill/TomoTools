using JuMP, NLPModels, Optimize, LinearOperators
using LSDescentMethods, Stopping, PyPlot, LineSearch

##Code de Hager-Zhang si on preconditionne en n'utilisant que les k dernieres
##directions, puisqu'on accepte la k-ieme quand on entre dans le subspace,
##meme si cette direction est consideree comme pas assez orthogonale!
##
##Version du 14 novembre 2017, qui tente de reproduire le code C.
##

include("subspace_iteration6.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)./(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function HZCG6(nlp :: AbstractNLPModel;
                    normal :: Bool=true,
                    m :: Int=11,
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
                    linesearch :: Function = Newarmijo_wolfe,
                    #linesearch :: Function = _hagerzhang2!,
                    scaling :: Bool = true,
                    print_h :: Bool = false,
                    print_h_iter :: Int64 = 1,
                    subspace :: Bool = true,
                    η₀ :: Float64=0.001,#Selon le code en C
                    η  :: Float64=0.4, #empiriquement, best is between (0,1)
                    η₁ :: Float64=0.9,#Selon le code en C
                    Θₖ :: Float64=1.0, #empiriquement, best is between [1,2)
                    assez_ortho :: Bool = true,
                    debugingQR :: Bool = false,
                    kwargs...)

        x = copy(nlp.meta.x0)
        n = nlp.meta.nvar
        if m > n
          m=n
        end

        xt = zeros(n);
        ∇ft = zeros(n);
        ∇fprec = zeros(n);
        s = zeros(x);
        y = zeros(n);


        f = obj(nlp, x)
        ft = Inf;

        iter = 1

        #∇f = grad(nlp, x)
        stp, ∇f = start!(nlp,stp,x)
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)

        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

        β = 0.0
        d = zeros(n);
        σₖ = 1.0

        Sₖ    = zeros(n, m)
        Z     = zeros(n, m)
        R     = zeros(m, m)
        Hₖ    = zeros(m, m)

        OK = true
        stalled_linesearch = false
        stalled_ascent_dir = false

        while (OK && !(optimal || tired || unbounded) ) #Boucle principale

            d = - ∇f + β * d
            slope = ∇f ⋅ d

            if iter == 1
                h = LineModel(nlp, x, d * σₖ)######TODO essayer de le mettre ligne 68 et d'enlever le if
            else
                h = Optimize.redirect!(h, x, d * σₖ)
            end#if iter

            debug = false

            t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft; verboseLS = verboseLS, debug = debug, kwargs...)


            t *= σₖ
            if linesearch in interfaced_algorithms
                ft = obj(nlp, x + t * d)
                nlp.counters.neval_obj += -1
            end#if in interfaced_algorithms

            ###Update of Sₖ
            if !normal #Si on veut utiliser QR fancy
              (Z,R) = MyZupdate(Sₖ,Z,R,iter,d,m)
              iter += 1
            else
              it       = mod(iter - 1,m) + 1
              Sₖ[:,it] = d
              Z,R      = qr(Sₖ[:,1:min(iter,m)])
              iter += 1
            end#if !normal

            xt = x + t * d#update du pas

            good_grad || (∇ft = grad!(nlp, xt, ∇ft))#calcul le gradient au nouveau point si pas deja fait

            st = xt - x
            yt = ∇ft - ∇f
            β = 0.0

            if scaling
                σₖ = BB(st,yt)
            end#if scaling

            ∇fNorm = norm(∇ft, Inf)

            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir
            assez_ortho = true

            (subspace && iter != 2) && (assez_ortho = ((1-η₀)*norm(∇ft).^2 >= norm((∇ft)'*Z[:,1:min(iter-1,m)]).^2))

            if !assez_ortho && (OK && !(optimal || tired || unbounded) )
                #Si la derniere direction n'est pas bonne, on la met quand meme
                #dans la matrice, on ne garde pas le nouveau point et on part
                #sur le subspace problem avec les vieilles valeurs (celles sans "t")

                (Z,R,xt,ft,∇ft,∇f,y,s,dt,iter,assez_ortho) = subspace_iteration6(nlp,x,f,∇f,m,Z[:,1:min(iter-1,m)],R,Sₖ,iter,assez_ortho,y,s;
                                                                                        η = η, normal=normal,linesearch=linesearch,kwargs...)

                optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
                OK = !stalled_linesearch & !stalled_ascent_dir
                

                dy = dt ⋅ y
                βₖ = ( (y ⋅ ∇ft) - Θₖ*( (y ⋅ y) * (dt ⋅ ∇ft)) / dy ) / dy
                ηₖ = η * ( (dt ⋅ ∇f) / (dt' * dt) )
                β  = max(βₖ, ηₖ)
                x = copy(xt)
                f = ft
                BLAS.blascopy!(n, ∇f, 1, ∇fprec, 1)
                BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
                d = copy(dt)

                if scaling
                    σₖ = BB(s,y)
                end#if scaling
            elseif assez_ortho && (OK && !(optimal || tired || unbounded) )

                dy = d ⋅ yt
                βₖ = ( (yt ⋅ ∇ft) - Θₖ * ( (yt ⋅ yt) * (d ⋅ ∇ft)) / dy ) / dy
                ηₖ = η * ( (d ⋅ ∇f) / (d' * d) )
                β  = max(βₖ, ηₖ)
                ###update des indices.
                x = copy(xt)
                f = ft
                BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
                s = st
                y = yt

            end#if assez_ortho

        end#while

        if optimal status = :Optimal
        elseif unbounded status = :Unbounded
        elseif stalled_linesearch status = :StalledLinesearch
        elseif stalled_ascent_dir status = :StalledAscentDir
        else status = :UserLimit
        end
        return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status)
end#function
