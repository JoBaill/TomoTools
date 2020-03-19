using JuMP, NLPModels, Optimize, LinearOperators
using LSDescentMethods, Stopping, PyPlot, LineSearch

##Code de Hager-Zhang si on preconditionne en n'utilisant que les k dernieres
##directions, puisqu'on accepte la k-ieme quand on entre dans le subspace,
##meme si cette direction est consideree comme pas assez orthogonale!
##
##Version du 14 novembre 2017, qui tente de reproduire le code C.
##

include("subspace_iteration6_1.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)/(yₖ⋅yₖ)
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
                    verboseLS :: Bool = true,
                    linesearch2 :: Function = Newarmijo_wolfe,#Linesearch du L-bfgs
                    #linesearch :: Function = Newarmijo_wolfe,#linesearch du L-CG
                    linesearch :: Function = _hagerzhang2!,#linesearch du L-CG
                    scaling :: Bool = true,
                    print_h :: Bool = false,
                    print_h_iter :: Int64 = 1,
                    subspace :: Bool = false,
                    η₀ :: Float64=0.01,#Selon le code en C
                    η  :: Float64=0.4, #empiriquement, best is between (0,1)
                    η₁ :: Float64=0.9,#Selon le code en C
                    Θₖ :: Float64=1.0, #empiriquement, best is between [1,2)
                    assez_ortho :: Bool = true,
                    debugingQR :: Bool = false,#parametre non utiliser presentement
                    kwargs...)

        x = copy(nlp.meta.x0)#point initial
        n = nlp.meta.nvar#dimension inititiale

        if m > n
          m=n
        end

        xt = zeros(n);#Initialisation des vecteurs necessaires
        ∇ft = zeros(n);
        ∇fprec = zeros(n);
        s = zeros(x);
        y = zeros(n);


        Sₖ    = zeros(n, m)#Initialisation des matrices necessaires
        Z     = zeros(n, m)
        R     = zeros(m, m)
        Hₖ    = zeros(m, m)

        β = 0.0#Initialisation des parametres necessaires
        d = zeros(n);
        σₖ = 1.0

        f = obj(nlp, x)
        #ft = Inf;

        iter = 1

        #∇f = grad(nlp, x)
        stp, ∇f = start!(nlp,stp,x)
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)

        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

        OK = true
        stalled_linesearch = false
        stalled_ascent_dir = false

        while (OK && !(optimal || tired || unbounded) ) #Boucle principale du gradient conjugue

if mod(iter,50) == 0
  println(f)
  println(nlp.counters)
  #error()
end#if
            d = - ∇f + β * d
            slope = ∇f ⋅ d

            if iter == 1
                h = LineModel(nlp, x, d * σₖ)
            else
                h = Optimize.redirect!(h, x, d * σₖ)
            end#if iter

            debug = false
println()
print_with_color(:green,"Pre linesearch $(nlp.counters)")
            t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft;mayterminate = false, verboseLS = verboseLS, debug = debug, kwargs...)
println()
print_with_color(:green,"Post linesearch $(nlp.counters)")

            t *= σₖ
            if linesearch in interfaced_algorithms
                ft = obj(nlp, x + t * d)
                nlp.counters.neval_obj += -1
            end#if in interfaced_algorithms

            ###Update of Sₖ
            # if !normal #Si on veut utiliser QR fancy
            #   (Z,R) = MyZupdate(Sₖ,Z,R,iter,d,m)
            #   iter += 1
            # else
              it       = mod(iter - 1,m) + 1
              Sₖ[:,it] = d
              Z,R      = qr(Sₖ[:,1:min(iter,m)])
              iter += 1
            #end#if !normal

            xt = x + t * d#update du pas

            good_grad || (∇ft = grad!(nlp, xt, ∇ft))#calcul le gradient au nouveau point si pas deja fait
println()
print_with_color(:cyan,"Post good_grad $(nlp.counters)")
            st = xt - x
            yt = ∇ft - ∇f
            β = 0.0

            if scaling
                σₖ = BB(st,yt)
            end#if scaling

            ∇fNorm = norm(∇ft, Inf)

            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir

            assez_ortho = true#on verifie l'orthogonalite avec: (1-η₀²) ⋝ ||∇f(x)ᵗ * Z|| / ||∇f(x)|| (equation 3.4)
            Normee = (norm((∇ft)'*Z[:,1:min(iter-1,m)]).^2) / (norm(∇ft).^2)
            (subspace && iter != 2) && (assez_ortho = ((1-η₀) >= Normee))
            #(subspace && iter != 2) && (assez_ortho = ((1-η₀)*norm(∇ft).^2 >= norm((∇ft)'*Z[:,1:min(iter-1,m)]).^2))
            #println(norm(xt-x))

            if !assez_ortho && (OK && !(optimal || tired || unbounded) )
                #Si la derniere direction n'est pas bonne, on la met quand meme
                #dans la matrice, on ne garde pas le nouveau point et on part
                #sur le subspace problem avec les vieilles valeurs (celles sans "t")

                I = min(iter-1,m)

                (Z,R,xt,ft,∇ft,∇f,yt,st,d,iter,assez_ortho) = subspace_iteration6(nlp,x,f,∇f,m,Z[:,1:I],R,Sₖ,iter,assez_ortho,y,s;
                                                                                        η = η, normal=normal,linesearch=linesearch2,kwargs...)
                                                                                        println(nlp.counters)

                optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
                OK = !stalled_linesearch & !stalled_ascent_dir
            
            end#if du subspace_iteration

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

                if scaling
                    σₖ = BB(s,y)
                end#if scaling

        end#while
        if optimal status = :Optimal
        elseif unbounded status = :Unbounded
        elseif stalled_linesearch status = :StalledLinesearch
        elseif stalled_ascent_dir status = :StalledAscentDir
        else status = :UserLimit
        end

        return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status)
end#function
