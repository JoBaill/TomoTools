#####Commence avec une backprojection comme image initiale#####

using NLPModels
using Optimize

import Compat.view

export LLModel, obj, grad, grad!,
       hess,objgrad,objgrad!,hprod,hprod!

import NLPModels.obj
import NLPModels.grad
import NLPModels.grad!
import NLPModels.hess
import NLPModels.objgrad
import NLPModels.objgrad!
import NLPModels.hprod
import NLPModels.hprod!

type TVLogTESTING <: AbstractNLPModel
    A::SparseMatrixCSC
    b::Vector
    LLs::AbstractNLPModel
    TVP::AbstractNLPModel
    lambda :: Real
    epsi :: Real
    i :: Array{Float64,1}
    j :: Array{Float64,1}
    meta :: NLPModelMeta
    counters :: Counters
    fgraph :: Array{Float64,1}
    Ggraph :: Array{Float64,1}
    Hugraph :: Array{Float64,1}
    proj_graph :: Array{Int64,1}
    values :: Array{Float64,1}
    Image :: Array{Float64,2}


    # Functions
    f :: Function
    g :: Function
    g! :: Function
    fg :: Function
    fg! :: Function
    H :: Function
    Hcoord :: Function
    Hp :: Function
    Hp! :: Function

end#Type

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

function TVLogtesting2(A::SparseMatrixCSC , b::Vector, lambda::Real, epsi::Real, Image;
    i::Array = zeros(size(A,2)),
    j::Array = zeros(size(A,2)),
    Fiability :: Function = fiability,
    f::Function = obj,
    g::Function = grad,
    g!::Function = grad!,
    fg::Function = objgrad,
    fg!::Function = objgrad!,
    #H::Function = LLhess,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = NotImplemented,#hprod,
    Hp!::Function = NotImplemented,#hprod!
    )

    nvar = size(A,2)
    #x0=vec(0.5*ones(nvar) + 0.01*rand(nvar))

    fgraph = []
    Ggraph = []
    Hugraph = []
    proj_graph = Array{Int64}([0])
    values = []

    m = size(A,2)
    srand(1234);
    #Xₖ = ((1.0 / m ) * sum(b)) * ones(m) + rand(m)
    Xₖ = A' * b;
    X = reshape(Xₖ, Int64(sqrt(length(Xₖ))), Int64(sqrt(length(Xₖ))))
    #LLS = MathProgNLPModel(LLs(b,b));

    TVP = MathProgNLPModel(TV(X,lambda;epsi=epsi));
    LLS = LogLmodel(A,b;Fiability = Fiability);
    lvar = zeros(nvar) + eps()
    uvar = Inf*ones(nvar)
    meta = NLPModelMeta(nvar,x0=Xₖ,lvar=lvar,uvar=uvar)

    return TVLogTESTING(A,b,LLS,TVP,lambda,epsi,i,j,meta,Counters(),fgraph,Ggraph,
                        Hugraph,proj_graph,values,Image,f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function obj(nlp :: TVLogTESTING, x :: Vector)

    f = obj(nlp.LLs,x) + obj(nlp.TVP,x)
    nlp.counters.neval_obj += 1
    push!(nlp.fgraph,f)
    nlp.LLs.fiability(x, nlp.Image, 1, nlp.proj_graph, nlp.values)
    return f
end

function grad!(nlp :: TVLogTESTING, x :: Vector, gradient :: Vector)
    gradient[:]  = grad(nlp.LLs,x) + grad(nlp.TVP,x)
    nlp.counters.neval_obj += 1
    push!(nlp.Ggraph,norm(gradient,Inf))
    nlp.LLs.fiability(x, nlp.Image, 2, nlp.proj_graph, nlp.values)
    return gradient
end

function grad(nlp :: TVLogTESTING, x :: Vector)
    g  = zeros(size(nlp.A,2))
    #nlp.counters.neval_grad += 1 #pas besoin car on utilise grad!
    return grad!(nlp,x,g)
end

# function objgrad!(nlp :: LLModel, x::Vector, gradient :: Vector)
#     nlp.counters.neval_obj += 1
#     nlp.counters.neval_grad += 1
#
#     Ax = nlp.A*x
#
#     f = obj(nlp.LLs,Ax)
#     g = zeros(gradient)#initialization
#
#     gg  = zeros(size(nlp.A,1))#for grad!(nlp.LLs)
#     g[:] = nlp.A' * grad!(nlp.LLs, Ax, gg)
#
#     f2,gradient[:] = objgrad!(nlp.TVP,x,gradient)
#
#     f = f+f2
#     gradient[:] = gradient + g
#
#     append!(nlp.fgraph,f)
#     append!(nlp.Ggraph,norm(gradient))
#
#     fiability(x, nlp.Image, 2, nlp.proj_graph, nlp.values)#techniquement on utilise res... voir LSQNLPModel.jl
#
#     return f,gradient
# end

function objgrad!(nlp :: TVLogTESTING, x :: Vector, gradient :: Vector)
    nlp.counters.neval_obj += 1 #besoin, on call pas obj et grad
    nlp.counters.neval_grad += 1
    #Ax = nlp.A*x + eps()
    #grad!(nlp,x,gradient)
    f, nlp.i = objgrad!(nlp.LLs, x, nlp.i)
    #nlp.i = nlp.A' * nlp.i
    g, nlp.j = objgrad!(nlp.TVP, x, nlp.j)
    nlp.LLs.fiability(x, nlp.Image, 2, nlp.proj_graph, nlp.values)#techniquement on utilise res... voir LSQNLPModel.jl
    gradient[:] = nlp.i + nlp.j
    push!(nlp.fgraph,f+g)
    push!(nlp.Ggraph,norm(gradient,Inf))
    return f+g, gradient
end


function objgrad(nlp :: TVLogTESTING, x::Vector)
    g  = zeros(size(nlp.A,2))
    return objgrad!(nlp,x,g)
end

# function hprod!(nlp :: LLModel, x :: Vector, u :: Vector, Hu :: Vector)
#   nlp.counters.neval_hprod += 1
#   Hu[:] = vec(u'*nlp.A'*hess(nlp.LLs,nlp.A*x)*(nlp.A)) + hprod!(nlp.TVP,x,u,Hu)
#   append!(nlp.Hugraph,norm(Hu))
#   return Hu
# end
#
# function hprod(nlp :: LLModel,x::Vector,u :: Vector)
#   Hu = zeros(size(nlp.A,2))
#   return hprod!(nlp, x, u, Hu)
# end
