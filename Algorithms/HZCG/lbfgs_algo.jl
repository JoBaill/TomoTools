export Newlbfgs

function Newlbfgs(nlp :: AbstractNLPModel;
                  atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                  max_eval :: Int=0,
                  itmax :: Int=5000,
                  verbose :: Bool=false,
                  verboseLS :: Bool = false,
                  mem :: Int=5,
                  linesearch :: Function = Newarmijo_wolfe,
                  kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    # PPP=nlp.counters.neval_obj + nlp.counters.neval_grad
    # println("")
    #   print_with_color(:cyan,string(PPP))
    # println("")
    H = InverseLBFGSOperator(n, mem, scaling=true)

    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    ϵ = atol + rtol * ∇fNorm
    max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
    iter = 0

    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

    while !(optimal || tired)
        d = - H * ∇f
#println("dₖₚ = $d")
        slope = BLAS.dot(n, d, 1, ∇f, 1)
        slope < 0.0 || error("Not a descent direction! slope = ", slope)
#println("slope = $slope")
        # Perform improved Armijo linesearch.
        h = C1LineFunction(nlp, x, d)

        t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=verboseLS; kwargs...)
println("")
println("t = $t")
println("")
#println("x = $x")
#println("xt = $xt")

        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, t, d, 1, xt, 1)
#println("x = $x")
#println("xt = $xt")
        # if iter == 2
        #   error()
        # end
#println("∇ft = $∇ft")


        good_grad || (∇ft = grad!(nlp, xt, ∇ft))

# PPP=nlp.counters.neval_obj + nlp.counters.neval_grad
# println("")
#   print_with_color(:magenta,string(PPP))
# println("")
#println("∇ft = $∇ft")
        # Update L-BFGS approximation.
        push!(H, t * d, ∇ft - ∇f)

        #print_with_color(:green,string(H * ones(n)))

        # Move on.
        x = xt
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)
        iter = iter + 1
println("")
print_with_color(:green,string(iter))
        optimal = ∇fNorm <= ϵ
        tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval
          # if iter==5
          #   error("done")
          # end
    end

    status = tired ? "maximum number of evaluations" : "first-order stationary"
    return (x, f, ∇fNorm, iter, optimal, tired, status)
end
