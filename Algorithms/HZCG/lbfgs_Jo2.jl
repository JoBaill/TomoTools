export Newlbfgs_Jo2
#include("armijo_wolfe6.jl")
#TODO: revoir le critère "Assez_ortho"

function Newlbfgs_Jo2(nlp :: AbstractNLPModel,subgrad,gₖₚ,gprec,f,Rinv;
                  stp :: TStopping=TStopping(),
                  verbose :: Bool=false,
                  verboseLS :: Bool = false,
                  mem :: Int=11,
                  linesearch :: Function = Newarmijo_wolfe,
                  #linesearch :: Function = _hagerzhang2!,
                  assez_ortho :: Bool = false,
                  η₁ :: Float64=0.999,
                  kwargs...)

    if verbose
        println("")
        println("")
        print_with_color(:green,"entering subspace LBFGS iterations")
        println("")
    end

    print_h=false
    n = nlp.meta.nvar
    x = copy(nlp.meta.x0)
    x_moins = copy(nlp.meta.x0)

    xt = zeros(n)

    ∇ft = zeros(n)
    d = zeros(n)
    t = 0.0

    H = InverseLBFGSOperator(n, mem, scaling=true)
    Hₖ = zeros(n,n)
    iter = 0

    ∇f = subgrad

    ###replaces "start" function as we already have the gradient###
    stp.optimality0 = stp.optimality_residual(∇f)
    stp.start_time  = time()
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    h_f = 0; h_g = 0; h_h = 0

    while (OK && !(optimal || tired || unbounded || assez_ortho))

        d = - H * ∇f

        slope = BLAS.dot(n, d, 1, ∇f, 1)

        if slope > 0.0
          stalled_linesearch =true
          verbose && @printf("  %8.1e", slope)
        else
          # Perform linesearch.
          if iter < 1
            h = LineModel(nlp, x, d)
          else
            h = Optimize.redirect!(h, x, d)
          end#if iter <1

          debug = false

          if print_h && (iter == print_h_iter)
            debug= true
            graph_linefunc(h, f, slope*scale;kwargs...)
          end#if print_h...

          h_f_init = copy(nlp.counters.neval_obj); h_g_init = copy(nlp.counters.neval_grad); h_h_init = copy(nlp.counters.neval_hprod)
          t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; verboseLS = verboseLS, kwargs...)
          h_f += copy(copy(nlp.counters.neval_obj) - h_f_init); h_g += copy(copy(nlp.counters.neval_grad) - h_g_init); h_h += copy(copy(nlp.counters.neval_hprod) - h_h_init)

          verbose && @printf("  %4d\n", nbk)

          BLAS.axpy!(n, t, d, 1, xt, 1)#update de l'itere xt

          good_grad || (∇ft = grad!(nlp, xt, ∇ft))#nouveau gradient dans ∇ft
          gprec[:] = copy(gₖₚ)
          gₖₚ[:] = copy(nlp.L_gx)

          # Update L-BFGS approximation.
          push!(H, t * d, ∇ft - ∇f)
          # Move on.
          x_moins = copy(x) #Pour le 2e alpha...
          x = copy(xt)

          f = ft
          BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
          ∇fNorm = BLAS.nrm2(n, ∇f, 1)

          iter = iter + 1

          verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

          assez_ortho = ((1-η₁) >= ((norm(Rinv' * ∇ft).^2)/(norm(gₖₚ).^2)) )
          optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
        end#if slope > 0.0
        OK = !stalled_linesearch & !stalled_ascent_dir

    end#while

    ###Saving H to use it as a preconditionner###
    Hₖ=full(H)
    verbose && @printf("\n")

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    if verbose

        print_with_color(:green,"exiting subspace LBFGS iterations")

        println("")
        println("")
    end

    return (x, x_moins, d, ∇ft, gₖₚ, gprec, Hₖ, tired, optimal, assez_ortho)

end
