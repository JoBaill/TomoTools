#include("BFGS6_gc.jl")
include("lbfgs_Jo.jl")
include("SNLPModel.jl")

function BB(sₖ,yₖ)
  if ( sₖ⋅yₖ > 0 )
    σₖ = (sₖ⋅yₖ)./(yₖ⋅yₖ)
  else
    σₖ = 1
  end
  return σₖ
end

function subspace_iteration6(nlp,
                            x,
                            f,
                            ∇f,
                            m,
                            Z,
                            R,
                            Sₖ,
                            iterb,
                            assez_ortho,
                            y,
                            s;
                            η  :: Float64=0.4,
                            normal :: Bool=true,
                            linesearch :: Function = Newarmijo_wolfe,
                            #solver :: Function = BFGS6_gc,
                            solver :: Function = Newlbfgs_Jo,
                            kwargs...)

        #gprec = ∇f
        gₖₚ = ones(∇f)
        gprec = ones(∇f)
        #∇ft = Array{Float64}(length(∇f))
        M = min(iterb-1,m) #Sets the dimension of D for later.
        #dt = Array{Float64}(length(∇f))
#println(M)
        #fabrication du NLPModels du probleme dans le sous-espace
#println()
#print_with_color(:green,"debug subspace iteration")
#println(size(Z))
#println(size(x))

 SPR=SNLPmodel(nlp,copy(x),Z)
#         function ϕ(α)
#           f=  obj(nlp, x + Z*α)
#           println("obj =$f")
#           return f
#         end
#
#         function ∇ϕ(α)
#           #global gprec[:] = copy(global gₖₚ)
#           G = grad(nlp, x + Z*α)
#           global gₖₚ[:] = copy(G)
#           return Z' * G #####voir si on pourrait sortir G = grad(nlp, x + Z*α)
#                 #####a chaque appel de cette fonction dans le LBFGS
#         end
#
#         function g!(α,g)
#           #global gprec[:] = copy(g)
#           G = grad(nlp, x + Z*α)
#           global gₖₚ[:] = copy(G)
#           g[:] = Z'*G#fabrique un effet de bord
#           println("gradient =$g")
#           return g
#         end
#
#         D = zeros(min(m,M))
#
# #        (iterb > M ) || ( D = zeros(M) )
# #println("D=$D")
#         SPR  = SimpleNLPModel(ϕ,D;g=∇ϕ,g! = g!)

#Version BOBOCHE
Z_avant = copy(Z)
x_avant = copy(x)
(α₁, α₂, t̂, d̂, ĝₖₚ, gₖₚ,gprec, Hₖ, tired, optimal, assez_ortho) = solver(SPR,Z,gₖₚ,gprec,∇f,f;mem=11,
                                                          kwargs...)
println("Z est correct: $(norm(Z_avant-Z)<0.001)")
println("x est correct: $(norm(x_avant-x)<0.001)")
# println("α₁, α₂, t̂, d̂, ĝₖₚ, gₖₚ = ")
# println("$α₁")
# println("$α₂")
# println("$t̂")
# println("$d̂")
# println("$ĝₖₚ")
# println("$gₖₚ")
# error("done")
#print_with_color(:red,"$(α₁-α₂)")
xprec = x + Z * α₂
x = x + Z * α₁
sₖ = x - xprec

#gprec = grad(nlp,xprec)
gₖₚ_avant = grad(nlp,x)
println("gₖₚ est correct: $(norm(gₖₚ_avant-gₖₚ)<0.001)")
#error("done")
fₖₚ = obj(nlp,x)

y = gₖₚ - gprec
#println(norm(y))
ŷ = Z' * y

dₗ = Z * d̂
σₖ = BB(sₖ,y)

βₖ = σₖ * ((y⋅gₖₚ - ŷ⋅ĝₖₚ) / (dₗ⋅y) - (norm(y)^2 - norm(ŷ)^2) * dₗ⋅gₖₚ / (dₗ⋅y).^2)
ηₖ = η * ( (sₖ' * gprec) / (dₗ' * y))

βₖ = max(βₖ, ηₖ)

dₚ   = -Z * (Hₖ - σₖ * eye(Hₖ)) * ĝₖₚ - σₖ * gₖₚ + βₖ * dₗ


slope = gₖₚ ⋅ dₚ

h = LineModel(nlp, x, dₚ)
gprec = copy(gₖₚ)
t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, fₖₚ, slope, gₖₚ; kwargs...)

#update de l'itere
xt   = x + t * dₚ

good_grad || (gₖₚ = grad!(nlp, xt, gₖₚ))
good_grad || (fₖₚ = obj(nlp,xt))


y    = gₖₚ - gprec

sₖ    = xt - x

normal = true

if !normal
  (Z,R) = MyZupdate(Sₖ,Z,R,iterb,dₚ,m)
else
          it       = mod(iterb-1,m)+1
          Sₖ[:,it] = dₚ
          Z,R=qr(Sₖ[:,1:M])
end
iterb += 1
# println("xt,fₖₚ,gₖₚ,gprec,y,sₖ,dₚ = ")
# println("$xt")
# println("$fₖₚ")
# println("$gₖₚ")
# println("$gprec")
# println("$y")
# println("$sₖ")
# println("$dₚ")
# error("done")

        return Z,R,xt,fₖₚ,gₖₚ,gprec,y,sₖ,dₚ,iterb,assez_ortho
end     #function







#         xprec = copy(x);
#         gprec = copy(∇f);
#
#         x = x + Z * D; # x₂plus
#         f = obj(nlp,x); #f₂plus
#         ∇f = grad(nlp,x); #g₂plus
#
#         s = s - xprec + x ;  #update suite a x₂=x₂plus
#         y = y - gprec + ∇f ;  #update suite a x₂=x₂plus
#         #TODO: d???
#         # On calcul maintenant dt a partir de la formule 4.3
#         ĝt = Z' * ∇f
#         ŷₖ = Z' * y
#
#         ηₖ = η * ( (s' * gprec) / (d * y))
#         βₖ = σₖ * ((y⋅∇f - ŷₖ⋅ĝt) / (d⋅y) - (norm(y)^2 - norm(ŷₖ)^2) * d⋅∇f / (d⋅y).^2)
#         βₖ = max(βₖ, ηₖ)
#
#         dt   = -Z * (Hₖ - σₖ * eye(Hₖ)) * ĝt - σₖ * ∇f + βₖᵖ * d
#
#
#
#         slope = ∇f ⋅ dt
#
#         h = LineModel(nlp, x, dt)
#
#         t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; kwargs...)
#
#         #update de l'itere
#         xt   = x + t * dt
#
#         good_grad || (∇ft = grad!(nlp, xt, ∇ft))
#         good_grad || (ft = obj(nlp,xt))
#
#
#         yₖ    = ∇ft - ∇f
#
#         sₖ    = xt - x
#
# normal = true
#
#         if !normal
#           (Z,R) = MyZupdate(Sₖ,Z,R,iterb,dt,m)
#
#         else
#                   it       = mod(iterb-1,m)+1
#                   Sₖ[:,it] = dt
#                   Z,R=qr(Sₖ[:,1:min(iterb,m)])
#         end
#         iterb += 1
#
#
