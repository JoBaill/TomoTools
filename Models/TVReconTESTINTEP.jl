#####Commence avec une image uniforme#####

using NLPModels
using Optimize

import Compat.view

export LASSOModel, obj, grad, grad!,
       hess,objgrad,objgrad!,hprod,hprod!

import NLPModels.obj
import NLPModels.grad
import NLPModels.grad!
import NLPModels.hess
import NLPModels.objgrad
import NLPModels.objgrad!
import NLPModels.hprod
import NLPModels.hprod!

type TVReconTESTING <: AbstractNLPModel
    A::SparseMatrixCSC
    b::Vector
    LSQ ::AbstractNLPModel
    TVP ::AbstractNLPModel
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

function TVRecontesting(A::SparseMatrixCSC , b::Vector, lambda::Real, epsi::Real, Image;
    i::Array = zeros(size(A,2)),
    j::Array = zeros(size(A,2)),
    Fiability :: Function = fiability,
    f::Function = obj,
    g::Function = grad,
    g!::Function = grad!,
    fg::Function = objgrad,
    fg!::Function = objgrad!,
    #H::Function = LASSOhess,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = hprod,
    Hp!::Function = hprod!
    )

#### nvar = size(LSQ.A,2)
#### x0=vec(0.5*ones(nvar) + 0.01*rand(nvar))

    fgraph = []
    Ggraph = []
    Hugraph = []
    proj_graph = Array{Int64}([0])
    values = []

    m = size(A,2)
    srand(1234);
    Xₖ = A' * b;
    #Xₖ = ((1.0 / m ) * sum(b)) * ones(m) + rand(m)
    X = reshape(Xₖ, Int64(sqrt(length(Xₖ))), Int64(sqrt(length(Xₖ))))

    LSQ = LSQNLPmodel(A,b;Fiability = Fiability);
    TVP = MathProgNLPModel(TV(X,lambda;epsi=epsi));

    meta = NLPModelMeta(m;x0=copy(Xₖ),lvar=zeros(m),uvar=Inf*ones(m))

    return TVReconTESTING(A,b,LSQ, TVP, lambda,epsi,i,j,meta,Counters(),fgraph,Ggraph,
                        Hugraph,proj_graph,values,Image,f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function obj(nlp :: TVReconTESTING, x :: Vector)
    f = LSQobj(nlp.LSQ,x) + obj(nlp.TVP,x)
    nlp.counters.neval_obj += 1
    push!(nlp.fgraph,f)
    nlp.LSQ.fiability(x, nlp.Image, 1, nlp.proj_graph, nlp.values)
    return f
end

function grad!(nlp :: TVReconTESTING, x :: Vector, gradient :: Vector)
    gradient[:] = LSQgrad!(nlp.LSQ,x,nlp.i) + grad!(nlp.TVP,x,nlp.j)
    nlp.counters.neval_obj += 1
    push!(nlp.Ggraph,norm(gradient,Inf))
    nlp.LSQ.fiability(x, nlp.Image, 2, nlp.proj_graph, nlp.values)
    return gradient
end

function grad(nlp :: TVReconTESTING, x :: Vector)
    g = zeros(nlp.meta.nvar)
    return grad!(nlp,x,g)
end

function objgrad!(nlp :: TVReconTESTING, x :: Vector, gradient :: Vector)

    f, nlp.i = LSQobjgrad!(nlp.LSQ, x, nlp.i)
    g, nlp.j = objgrad!(nlp.TVP,x,nlp.j)
    gradient[:] = nlp.i + nlp.j

    nlp.LSQ.fiability(x, nlp.Image, 2, nlp.proj_graph, nlp.values)

    return f+g, gradient
end


function objgrad(nlp :: TVReconTESTING, x :: Vector)
    g = zeros(nlp.meta.nvar)
    return objgrad!(nlp,x,g)
end

function hprod!(nlp :: TVReconTESTING, x :: Vector, v :: Vector, Hv :: Vector)
    nlp.counters.neval_hprod += 1
    Hv[:] = (nlp.A' * (nlp.A*v)) + hprod!(nlp.TVP,x,v,Hv)
    return Hv
end

function hprod(nlp :: TVReconTESTING, x :: Vector, v :: Vector)
    #nlp.counters.neval_hprod += 1 #pas besoin, on usitilise hprod!
    Hv = zeros(size(nlp.A,2))
    return hprod!(nlp,x,v,Hv)
end
