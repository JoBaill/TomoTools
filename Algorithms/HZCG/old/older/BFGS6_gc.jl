using JuMP
using NLPModels
using Optimize
using LinearOperators

#include("armijo_wolfe6.jl")
# include("arwheadJo.jl");
# prob1=MathProgNLPModel(arwheadJo());
#
# (xₖ,f,gₖNorm,iter)=BFGS_Jo(prob1)

export BFGS6_gc

function BFGS6_gc(nlp :: AbstractNLPModel,
              Z;
              atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
              max_eval :: Int=0,
              ϵ :: Float64=1.0e-6,
              assez_ortho :: Bool = false,
              η₁ :: Float64=0.9,
              kwargs...
              )
  x = copy(nlp.meta.x0)  #valeur initiale
  n = nlp.meta.nvar  #nombre de variable

  xt = Array{Float64}(n)  #Initialisation
  yₖ = Array{Float64}(n)
  ∇ft = Array{Float64}(n)
  d = Array{Float64}(n)
  dₚ = Array{Float64}(n)

  f = obj(nlp, x)  #valeur de depart
  ∇f = grad(nlp, x)
  H = eye(n,n)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  # ϵ = atol + rtol * ∇fNorm  #fabrication des conditions d'arret
  max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
  iter = 0

  optimal = ∇fNorm <= ϵ
  tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

  while !(optimal || tired || assez_ortho)

    dₚ = copy(d)
    d  = -H * ∇f  #calcul de la direction

    slope = BLAS.dot(n, d, 1, ∇f, 1)  #calcul du pas

    slope < 0.0 || error("Not a descent direction! slope = ", slope)

    # Perform improved Armijo linesearch.
    h = LineModel(nlp, x, d)
    t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = Newarmijo_wolfe(h, f, slope, ∇ft; kwargs...)


    xt = copy(x)
    xt = xt + t * d
#     println("")
# print_with_color(:cyan,"good_grad = $good_grad")
# println("")
    good_grad || (∇ft = grad!(nlp, xt, ∇ft))

      ∇fNorm = BLAS.nrm2(n, ∇ft, 1)
      iter = iter + 1

#      assez_ortho = ((1-η₁^2)*norm(Z*∇ft).^2 >= norm((Z*∇ft)'*Z).^2)

      assez_ortho = ((1-η₁^2)*norm(∇ft).^2 >= norm((∇ft)'*Z).^2)
      optimal = ∇fNorm <= ϵ
      tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

    # Update BFGS approximation.
    yₖ  = ∇ft - ∇f

    sₖ  = xt - x

    if !(optimal || tired || assez_ortho)

      ρₖ  = 1 / (dot(vec(yₖ),vec(sₖ)) )

      Vₖ  = eye(n,n) - (ρₖ * (yₖ * (sₖ)'))

      H = (dot(yₖ,sₖ)/dot(H*yₖ,yₖ)) * H

      H   = ((Vₖ)' * H * Vₖ) + ρₖ*(sₖ * (sₖ)')

      # Move on.
      x = xt
      BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
    end#if

  end#while
  println("")
  print_with_color(:red,"BFGS OPTIMAL = $optimal")
  println("")
  return (xt,H,tired, optimal,assez_ortho)
end#function
