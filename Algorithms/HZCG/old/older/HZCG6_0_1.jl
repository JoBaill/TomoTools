export HZCG

using JuMP
using NLPModels
using Optimize
using LinearOperators
using LSDescentMethods
using Stopping
using PyPlot
using LineSearch

#include("armijo_wolfe6.jl")
include("QRupdate.jl")
include("lbfgs_algo.jl")
include("subspace_iteration.jl")
include("preconditionning_step.jl")
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

function HZCG(nlp :: AbstractNLPModel;
                    normal :: Bool=true,
                    m :: Int=11,
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
#                    linesearch :: Function = Newarmijo_wolfe,
                    linesearch :: Function = _hagerzhang2!,
                    scaling :: Bool = true,
                    print_h :: Bool = false,
                    print_h_iter :: Int64 = 1,
                    subspace :: Bool = true,
                    η₀ :: Float64=0.001,#Selon le code en C
                    η  :: Float64=0.4, #empiriquement, best is between (0,1)
                    η₁ :: Float64=0.9,#Selon le code en C
                    Θₖ :: Float64=1.0, #empiriquement, best is between [1,2)
                    assez_ortho :: Bool = true,
                    kwargs...)

    x = copy(nlp.meta.x0)
    println(x[1:10])
    n = nlp.meta.nvar

# '''
#     if m > n
#       #println("dimension of the problem = ", n)
#       error("need memory to be <= than the dimension of the problem")
#     end
# '''

    xt = Array{Float64}(n);
    ∇ft = Array{Float64}(n);

    f = obj(nlp, x)

    iter = 1

    #∇f = grad(nlp, x)
    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    if n <= 2
      verbose && @printf("%4s  %8s  %11s %17s  %13s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "x", "∇f'd", "bk","t","σₖ","h'(t)","t_original")
      verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm,x)
    else
      verbose && @printf("%4s  %8s  %11s %8s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "∇f'd", "bk","t","σₖ","h'(t)","t_original")
      verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
    end

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    β = 0.0
    d = zeros(∇f)
    σₖ = 1.0

    Sₖ    = zeros(n, m)
    Z     = zeros(n, m)
    R     = zeros(m, m)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    h_f = 0; h_g = 0; h_h = 0

    while (OK && !(optimal || tired || unbounded) )
        d = - ∇f + β*d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
          #stalled_ascent_dir = true
          d = - ∇f
          slope =  ∇f⋅d
        end
        # #else
        verbose && @printf(" %10.1e", slope*σₖ)

        # Perform linesearch.
        if iter == 1
          h = LineModel(nlp, x, d*σₖ)
        else
          h = Optimize.redirect!(h, x, d*σₖ)
        end

        debug = false

        if print_h && (iter == print_h_iter)
          debug= true
          graph_linefunc(h, f, slope*σₖ;kwargs...)
        end

        verboseLS && println(" ")

        h_f_init = copy(nlp.counters.neval_obj);
        h_g_init = copy(nlp.counters.neval_grad);
        h_h_init = copy(nlp.counters.neval_hprod);

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * σₖ, ∇ft; verboseLS = verboseLS, debug = debug, kwargs...)
        println("")
        print_with_color(:red,"normal t = $t")
        println("")
        h_f += copy(copy(nlp.counters.neval_obj) - h_f_init);
        h_g += copy(copy(nlp.counters.neval_grad) - h_g_init);
        h_h += copy(copy(nlp.counters.neval_hprod) - h_h_init)

        if linesearch in interfaced_algorithms
          ft = obj(nlp, x + (t*σₖ)*d)
          nlp.counters.neval_obj += -1
        end

        t *= σₖ
        if verboseLS
          verbose && print("\n")
        else
          verbose && @printf("  %4d  %8e  %8e %8e  %8e\n", nbk, t, σₖ,grad(h,t),t_original*σₖ)
        end

        xt = x + t*d

        good_grad || (∇ft = grad!(nlp, xt, ∇ft))
        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = 0.0
"""

        println("ALLO")
       if (∇ft⋅∇f) < 0.2 * (∇ft⋅∇ft)   # Powell restart
            β = CG_formula(∇f,∇ft,s,d)
        end

"""
        if scaling
            σₖ = BB(s,y)
        end


"""
#TEST: LES PLACER DANS LE ELSE PLUS LOIN
        x = xt
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

"""

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = norm(∇ft, Inf)
#        iter = iter + 1


        if n <= 2
          verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm, x)
        else
          verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
        end


        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)

        OK = !stalled_linesearch & !stalled_ascent_dir

        subspace && (assez_ortho = ((1-η₀^2)*norm(∇ft).^2 >= norm((∇ft)'*Z).^2))
println("")
print_with_color(:green,"nlp.counters.neval_grad normal = $(nlp.counters.neval_grad)")
println("")
        if !assez_ortho
          println("")
          print_with_color(:cyan,"$iter")
        # x = copy(xt)
        # f  = copy(ft)
        # ∇f = copy(∇ft)

        print_with_color(:yellow,"f=$f")

          (Z,R,Hₖ,xt,ft,∇ft,gₖ,yₖ,sₖ,dₖ,iter) = subspace_iteration(nlp,x,f,∇f,m,Z,R,Sₖ,iter;normal =normal,linesearch=linesearch,kwargs...)
println("")
print_with_color(:green,"nlp.counters.neval_grad subspace = $(nlp.counters.neval_grad)")
println("")
print_with_color(:yellow,"ft=$ft")
println("")
print_with_color(:yellow,"xt=$xt")
println("")
print_with_color(:yellow,"g=$(norm(∇ft))")
          optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
          OK = !stalled_linesearch & !stalled_ascent_dir

          if (OK && !(optimal || tired || unbounded) )
            x = copy(xt)
            f  = copy(ft)
            ∇f = copy(∇ft)
            (Z,R,xt,ft,∇ft,yₖ,sₖ,dₖ,βₖᵖ,iter) = preconditionning_step(nlp,x,f,∇f,gₖ,m,Z,R,Sₖ,Hₖ,yₖ,sₖ,dₖ,iter;normal =normal,η=η,Θₖ=Θₖ,linesearch=linesearch,kwargs...)
            optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,xt,ft,∇ft)
            OK = !stalled_linesearch & !stalled_ascent_dir
          end#if
        else #on prepare le prochain itere
          dy = d⋅y
          βₖ    = (y⋅∇ft)/(dy) - Θₖ*((y⋅y)*(d⋅∇ft))/((dy).^2)
          ηₖ    = η * ( (d⋅∇f) / (d'*d) )
          β     = max(βₖ, ηₖ)

          x = xt
          f = ft
          BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)



if normal
           (Z,R) = MyZupdate(Sₖ,Z,R,iter,d)
          QR=TestQR(Z,R,Sₖ,d,iter)
          println("QR = $QR")
          ZR = TestZR(Z,R,Sₖ;normee=true)
          println("ZR = $ZR")
            print_with_color(:red,"rank(S)=$(rank(Sₖ))")
#           if !(ZR[1])
#             (Z,R)=qr(Sₖ)
#           end"""
else
it       = mod(iter-1,m)+1
Sₖ[:,it] = d
Z,R=qr(Sₖ)
end

          iter +=1

        end#if

    end
    verbose && @printf("\n")

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h_f, h_g, h_h)
end
