#include("BFGS6_gc.jl")
include("lbfgs_Jo2.jl")
include("SNLPModel.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)/(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function subspace_iteration7(nlp :: AbstractNLPModel,
                            x :: Array{Float64,1},
                            f :: Real,
                            subgrad :: Array{Float64,1},#Z'*∇ft
                            R :: Array{Float64,2},
                            Sₖ :: Array{Float64,2},
                            iterb :: Int64,
                            assez_ortho :: Bool,
                            y :: Array{Float64,1},
                            s :: Array{Float64,1},
                            σ :: Real,
                            m :: Int64;
                            verbose :: Bool=false,
                            verboseLS :: Bool = false,
                            η  :: Float64=0.4,
                            normal :: Bool=true,
                            linesearch :: Function = Newarmijo_wolfe,
                            #solver :: Function = BFGS6_gc,
                            solver :: Function = Newlbfgs_Jo2,
                            kwargs...)

    gₖₚ = ones(x)
    gprec = ones(x)
    dₚ = zeros(x)
    ŷ  = zeros(m)
    Rinv = inv(triu(copy(R)))
    S_petit = Sₖ[:, 1:min(iterb - 1,m)]

    M = min(iterb - 1, m) #Sets the dimension of D for later.
    w = copy(x)###Regle temporairement le probleme d'effet de bord indesirable###

    ###Creating the subproblem and solving it via L-BFGS
    SPR=SNLPmodel(nlp, w, S_petit)
    stp2 = TStopping(;atol = 1.0e-6, rtol = 1.0e-8)

    (α₁, α₂, d̂, ĝₖₚ, gₖₚ,gprec, Hₖ, tired, optimal, assez_ortho) = solver(SPR,subgrad,gₖₚ,gprec,f,Rinv;
                                                                    verbose=verbose,
                                                                    verboseLS=verboseLS,
                                                                    stp=stp2,
                                                                    mem=11,
                                                                    kwargs...)

    if norm(d̂) <= eps()
      error("Precision insuffisante du lbfgs")
    end

    ###Starting the preconditionning step###

    xprec = x + (S_petit * α₂)#could be replace by Xα.
    x = x + (S_petit * α₁)#could be replace by Xα.

    s = x - xprec

    fₖₚ = obj(nlp,x)

    y = gₖₚ - gprec
    ŷ = Rinv' * (S_petit' * y)

    dₚ = (S_petit * (Rinv * d̂))
    σₖ = BB(s,y)

    dly = dₚ⋅y

    βₖ = σₖ * ((y⋅gₖₚ - ŷ⋅ĝₖₚ) / (dly) - (norm(y)^2 - norm(ŷ)^2) * dₚ⋅gₖₚ / (dly).^2)
    ηₖ = η * ( (s' * gprec) / (dly))

    βₖ = max(βₖ, ηₖ)

    dₚ   = -S_petit* (Rinv * ((Hₖ - σₖ * eye(Hₖ)) * ĝₖₚ)) - σₖ * gₖₚ + βₖ * dₚ

    slope = gₖₚ ⋅ dₚ

    verboseLS && println(" ")

    h = LineModel(nlp, x, dₚ)
    gprec = copy(gₖₚ)
    t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, fₖₚ, slope, gₖₚ; kwargs...)

    t *= σ
    if verboseLS
      verbose && print("\n")
    end


    ###update de l'itere###
    xt   = x + t * dₚ

    ###needed if using _hagerzhang2 cause it fixes f and g at NaN ###
    good_grad || (gₖₚ = grad!(nlp, xt, gₖₚ))
    good_grad || (fₖₚ = obj(nlp,xt))

    y    = gₖₚ - gprec
    s    = xt - x

    ###update of Z###

    if iterb > m #qrupdate (rank 1 update)
        R = qrupdate3!(Sₖ, R, dₚ, iterb)
    else #qraddcol (updating R while adding columns to Sₖ)
        R = qraddcol(S_petit,R,dₚ)
        Sₖ[:,iterb] = dₚ
    end#if !normal

    iterb += 1

    return R,xt,fₖₚ,gₖₚ,gprec,y,s,dₚ,iterb,assez_ortho

end#function
