#include("BFGS6_gc.jl")
include("lbfgs_Jo.jl")
include("SNLPModel.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)/(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function subspace_iteration6(nlp :: AbstractNLPModel,
                            x :: Array{Float64,1},
                            f :: Real,
                            ∇f :: Array{Float64,1},
                            m:: Int64,
                            Z :: Array{Float64,2},
                            R :: Array{Float64,2},
                            Sₖ :: Array{Float64,2},
                            iterb :: Int64,
                            assez_ortho :: Bool,
                            y :: Array{Float64,1},
                            s :: Array{Float64,1},
                            σ :: Real;
                            verbose :: Bool=false,
                            verboseLS :: Bool = false,
                            η  :: Float64=0.4,
                            normal :: Bool=true,
                            linesearch :: Function = Newarmijo_wolfe,
                            #solver :: Function = BFGS6_gc,
                            solver :: Function = Newlbfgs_Jo,
                            kwargs...)

#print_with_color(:red,"verbose = $verbose, verboseLS = $verboseLS")
    gₖₚ = ones(∇f)
    gprec = ones(∇f)
    dₚ = zeros(∇f)
    ŷ  = zeros(m)

    M = min(iterb-1,m) #Sets the dimension of D for later.
    w = copy(x)###Regle temporairement le probleme d'effet de bord indesirable###

    ###Creating the subproblem and solving it via L-BFGS
    #SPR=SNLPmodel(nlp,copy(x),Z)
    SPR=SNLPmodel(nlp,w,Z)
    stp2 = TStopping(;atol = 1.0e-6,rtol = 1.0e-8)

    (α₁, α₂, t̂, d̂, ĝₖₚ, gₖₚ,gprec, Hₖ, tired, optimal, assez_ortho) = solver(SPR,Z,gₖₚ,gprec,∇f,f;
                                                                            verbose=verbose,
                                                                            verboseLS=verboseLS,
                                                                            stp=stp2,
                                                                            mem=11,
                                                                            kwargs...)

    if norm(d̂) <= eps()
      error("Precision insuffisante du lbfgs")
    end

    ###Starting the preconditionning step###

    xprec = x + Z * α₂#could be replace by Xα.
    x = x + Z * α₁#could be replace by Xα.

    s = x - xprec

    fₖₚ = obj(nlp,x)

    y = gₖₚ - gprec
    ŷ = Z' * y

    dₚ = Z * d̂
    σₖ = BB(s,y)

    dly = dₚ⋅y

    βₖ = σₖ * ((y⋅gₖₚ - ŷ⋅ĝₖₚ) / (dly) - (norm(y)^2 - norm(ŷ)^2) * dₚ⋅gₖₚ / (dly).^2)
    ηₖ = η * ( (s' * gprec) / (dly))

    βₖ = max(βₖ, ηₖ)

    dₚ   = -Z * (Hₖ - σₖ * eye(Hₖ)) * ĝₖₚ - σₖ * gₖₚ + βₖ * dₚ

    slope = gₖₚ ⋅ dₚ

    verboseLS && println(" ")

    h = LineModel(nlp, x, dₚ)
    gprec = copy(gₖₚ)
    t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, fₖₚ, slope, gₖₚ; kwargs...)

    t *= σ
    if verboseLS
      verbose && print("\n")
    # else
    #
    #   verbose && @printf("  %4d  %8e  %8e %8e  %8e\n", nbk, t, σ,grad(h,t),t_original*σ)
    #

    end


    ###update de l'itere###
    xt   = x + t * dₚ
#    println("xt = $xt")
    ###needed if using _hagerzhang2 cause it fixes f and g at NaN ###
    good_grad || (gₖₚ = grad!(nlp, xt, gₖₚ))
    good_grad || (fₖₚ = obj(nlp,xt))

    y    = gₖₚ - gprec
    s    = xt - x

    ###update of Z###
    #normal = true

    if !normal
        (Z,R) = MyZupdate(Sₖ,Z,R,iterb,dₚ,m)
    else
        it       = mod(iterb - 1,m) + 1
        Sₖ[:,it] = dₚ
        Z,R = qr(Sₖ[:,1:M])
    end#if !normal

    iterb += 1

        return Z,R,xt,fₖₚ,gₖₚ,gprec,y,s,dₚ,iterb,assez_ortho
end#function
