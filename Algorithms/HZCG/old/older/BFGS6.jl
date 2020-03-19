using JuMP
using NLPModels
using Optimize
using LinearOperators

include("armijo_wolfe6.jl")
# include("arwheadJo.jl");
# prob1=MathProgNLPModel(arwheadJo());
#
# (xₖ,f,gₖNorm,iter)=BFGS_Jo(prob1)

export BFGS

function BFGS(nlp :: AbstractNLPModel;
              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
              max_eval :: Int=0,
              ϵ :: Float64=1.0e-5,
              kwargs...
              )
  x = copy(nlp.meta.x0)  #valeur initiale
  n = nlp.meta.nvar  #nombre de variable

  xt = Array{Float64}(n)  #Initialisation
  yₖ = Array{Float64}(n)
  ∇ft = Array{Float64}(n)

  f = obj(nlp, x)  #valeur de depart
  ∇f = grad(nlp, x)
  H = eye(n,n)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  # ϵ = atol + rtol * ∇fNorm  #fabrication des conditions d'arret
  max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
  iter = 0

  optimal = ∇fNorm <= ϵ
  tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

  while !(optimal || tired)


    d  = -H * ∇f  #calcul de la direction

    slope = BLAS.dot(n, d, 1, ∇f, 1)  #calcul du pa

    slope < 0.0 || error("Not a descent direction! slope = ", slope)

    # Perform improved Armijo linesearch.
    h = LineModel(nlp, x, d)
    t, good_grad, f, nbk, nbW = Newarmijo_wolfe(h, f, slope, ∇ft)


    xt = copy(x)
    xt = xt + t * d

    good_grad || (∇ft = grad!(nlp, xt, ∇ft))

    # Update BFGS approximation.
    yₖ  = ∇ft - ∇f

    sₖ  = xt - x

    ρₖ  = 1 / (dot(vec(yₖ),vec(sₖ)) )

    Vₖ  = eye(n,n) - (ρₖ * (yₖ * (sₖ)'))

    H = (dot(yₖ,sₖ)/dot(H*yₖ,yₖ)) * H

    H   = ((Vₖ)' * H * Vₖ) + ρₖ*(sₖ * (sₖ)')

    # Move on.
    x = xt
    BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    iter = iter + 1

    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval


  end
  return (x, f, ∇fNorm, iter,tired,optimal)
end
