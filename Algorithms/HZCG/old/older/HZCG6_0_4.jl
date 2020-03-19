using JuMP, NLPModels, Optimize, LinearOperators
using LSDescentMethods, Stopping, PyPlot, LineSearch

##Code de Hager-Zhang si on preconditionne en utilisant que les k dernieres
##directions, puisqu'on accepte la k-ieme quand on entre dans le subspace,
##meme si cette direction est consideree comme pas assez orthogonale!
##
##
##

include("QRupdate.jl")
include("lbfgs_algo.jl")
include("subspace_iteration4.jl")
include("preconditionning_step4.jl")
include("BFGS6_gc.jl")
include("TestMultiple.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)./(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function HZCG4(nlp :: AbstractNLPModel;
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
                    η₀ :: Float64=0.2,#Selon le code en C
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
println(n)
        xt = zeros(n);
        ∇ft = zeros(n);
        ∇fprec = zeros(n);

        f = obj(nlp, x)
        ft = Inf;

        iter = 1

        #∇f = grad(nlp, x)
        stp, ∇f = start!(nlp,stp,x)
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)

        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
println("iter apres stop = $iter")
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

println("")
print_with_color(:red,"objective = $(obj(nlp,x))")
println("")

if debugingQR
  if iter > 1
      i    = mod(iter - 1,m) + 1
      if iter > m
        QR = (norm(Z*R - Sₖ[:,1:i]) < 0.0001)
      else
        QR = (norm(Z*R - Sₖ[:,1:i-1]) < 0.0001)
      end#if else
      println("QR = $QR")
    end#if iter..
end#if
            d = - ∇f + β * d
            slope = ∇f ⋅ d
println("")
print_with_color(:cyan,"slope dans GC normal: $slope, f = $f")

            if iter == 1
                h = LineModel(nlp, x, d * σₖ)######TODO essayer de le mettre ligne 68 et d'enlever le if
            else
                h = Optimize.redirect!(h, x, d * σₖ)
            end#if iter

            debug = false
println("")
print_with_color(:green,"Test de direction dans le GC normal")
TestDirectionSubspace(Z,d,iter,m)

            t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft; verboseLS = verboseLS, debug = debug, kwargs...)

            if linesearch in interfaced_algorithms
                ft = obj(nlp, x + (t * σₖ) * d)
                nlp.counters.neval_obj += -1
            end#if in interfaced_algorithms

            t *= σₖ

            xt = x + t * d#update du pas

            good_grad || (∇ft = grad!(nlp, xt, ∇ft))#calcul le gradient au nouveau point si pas deja fait

            s = xt - x
            y = ∇ft - ∇f
            β = 0.0

            if scaling
                σₖ = BB(s,y)
            end#if scaling

            ∇fNorm = norm(∇ft, Inf)

            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir
            assez_ortho = true
print_with_color(:yellow,"iter = $iter")
            (subspace && iter != 1) && (assez_ortho = ((1-η₀^2)*norm(∇ft).^2 >= norm((∇ft)'*Z[:,1:min(iter-1,m)]).^2))

            if !assez_ortho && (OK && !(optimal || tired || unbounded) )
                #Si la derniere direction n'est pas bonne, on la met quand meme
                # dans la mtrice, on ne garde pas le nouveau point et on part
                #sur le subspace problem avec les vieilles valeurs (celles sans "t")

                it       = mod(iter - 1,m) + 1
                Sₖ[:,it] = d
                Z,R      = qr(Sₖ[:,1:min(iter,m)])

                (Z,R,Hₖ,xt,ft,∇ft,∇f,y,sₖ,dt,iter,assez_ortho) = subspace_iteration4(nlp,x,f,∇f,m,Z[:,1:min(iter,m)],R,Sₖ,iter,assez_ortho;normal=normal,linesearch=linesearch,kwargs...)
#error("subspace")
println("size(Z) = $(size(Z))")
println("size(R) = $(size(R))")
println("size(Sₖ) = $(size(Sₖ[:,1:min(iter,m)]))")
println("norm de la dif = $(norm(Sₖ[:,1:min(iter-1,m)] - Z*R))")
println("test direction")
TestDirectionSubspace(Z,d,iter,m)
println("obj = $(obj(nlp,xt))")
#error("je suis tanner")
                optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
                OK = !stalled_linesearch & !stalled_ascent_dir

                dy = dt ⋅ y
                βₖ = ( (y ⋅ ∇ft) - Θₖ*( (y ⋅ y) * (dt ⋅ ∇ft)) / dy ) / dy
                ηₖ = η * ( (dt ⋅ ∇f) / (dt' * dt) )
                β  = max(βₖ, ηₖ)
                #β  = 0
                x = copy(xt)
                f = ft
                BLAS.blascopy!(n, ∇f, 1, ∇fprec, 1)
                BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
                d = copy(dt)
                if (OK && !(optimal || tired || unbounded) )
                    (Z,R,x,f,∇f,y,s,d,β,iter)=preconditionning_step4(nlp,x,f,∇f,∇fprec,m,Z,R,Sₖ,Hₖ,y,sₖ,d,iter;linesearch=linesearch,kwargs...)

                    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
                    OK = !stalled_linesearch & !stalled_ascent_dir

                end#if du preconditionning_step
                if scaling
                    σₖ = BB(s,y)
                end#if scaling
            elseif assez_ortho && (OK && !(optimal || tired || unbounded) )
                dy = d ⋅ y
                βₖ = ( (y ⋅ ∇ft) - Θₖ * ( (y ⋅ y) * (d ⋅ ∇ft)) / dy ) / dy
                ηₖ = η * ( (d ⋅ ∇f) / (d' * d) )
                β  = max(βₖ, ηₖ)
                x = xt
                f = ft
                BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

                if !normal #Si on veut utiliser QR fancy
                    (Z,R) = MyZupdate(Sₖ,Z,R,iter,d,m)
                    iter += 1
                else
                    it       = mod(iter - 1,m) + 1
                    Sₖ[:,it] = d
                    Z,R      = qr(Sₖ[:,1:min(iter,m)])
                    iter += 1
                end#if !normal
            end#if assez_ortho


if iter != 2
println("rank(Z) = $(rank(Z))")
end#if
        end#while

        if optimal status = :Optimal
        elseif unbounded status = :Unbounded
        elseif stalled_linesearch status = :StalledLinesearch
        elseif stalled_ascent_dir status = :StalledAscentDir
        else status = :UserLimit
        end
        return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status)
end#function
