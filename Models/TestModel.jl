#####Commence avec une image uniforme#####

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

type TestModel <: AbstractNLPModel
    b::Vector
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

function modeltesting(b::Vector, epsi::Real;
    i::Array = zeros(size(b,1)),
    j::Array = zeros(size(b,1)),
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

    nvar = size(b,1)
    x0 = vec(0.5*ones(nvar) + 0.01*rand(nvar))
    meta = NLPModelMeta(nvar;x0=x0)

    fgraph = []
    Ggraph = []
    Hugraph = []
    proj_graph = Array{Int64}([0])
    values = []

    srand(1234);

    return TestModel(b,epsi,i,j,meta,Counters(),fgraph,Ggraph,
                        Hugraph,proj_graph,values,f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function obj(nlp :: TestModel, x :: Vector)

    f = ones(x)' * (nlp.b .* x.^2)
    nlp.counters.neval_obj += 1
    push!(nlp.fgraph,f)
    return f
end

function grad!(nlp :: TestModel, x :: Vector, gradient :: Vector)
    gradient[:]  = 2 * (nlp.b .* x)
    nlp.counters.neval_obj += 1
    push!(nlp.Ggraph,norm(gradient,Inf))
    return gradient
end

function grad(nlp :: TestModel, x :: Vector)
    g  = zeros(size(nlp.b,1))
    #nlp.counters.neval_grad += 1 #pas besoin car on utilise grad!
    return grad!(nlp,x,g)
end

function objgrad!(nlp :: TestModel, x :: Vector, gradient :: Vector)
    nlp.counters.neval_obj += 1 #besoin, on call pas obj et grad
    nlp.counters.neval_grad += 1

    f=obj(nlp,x)
    gradient[:] =grad!(nlp,x,gradient)

    return f, gradient
end

function objgrad(nlp :: TestModel, x::Vector)
    g  = zeros(size(nlp.b,1))
    return objgrad!(nlp,x,g)
end
