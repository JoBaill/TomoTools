using NLPModels
using Optimize

import Compat.view

export LSQNLPModel, LSQobj, LSQgrad, LSQgrad!,
       LSQhess,LSQobjgrad,LSQobjgrad!, LSQhprod, LSQhprod!

type LSQNLPModel <: AbstractNLPModel
    A::SparseMatrixCSC
    b::Vector
    # lambda::Real
    # epsi::Real
    meta :: NLPModelMeta
    counters :: Counters
    fiability :: Function

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

function LSQNLPmodel(A::SparseMatrixCSC, b::Vector;
    Fiability :: Function = fiability,
    f::Function = LSQobj,
    g::Function = LSQgrad,
    g!::Function = LSQgrad!,
    fg::Function = LSQobjgrad,
    fg!::Function = LSQobjgrad!,
    H::Function = LSQhess,
    Hcoord::Function = NotImplemented,
    Hp::Function = LSQhprod,
    Hp!::Function = LSQhprod!
    )

    nvar = size(A,2)

    meta = NLPModelMeta(nvar)

    return LSQNLPModel(A,b,meta,Counters(),Fiability,f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function LSQobj(nlp :: LSQNLPModel, x :: Vector)
    res = nlp.A*x-nlp.b
    nlp.counters.neval_obj += 1
    return 0.5*dot(res,res)
end

function LSQgrad!(nlp :: LSQNLPModel, x :: Vector, gradient :: Vector)
    res = nlp.A*x-nlp.b
    gradient[:]= (nlp.A)'*res
    nlp.counters.neval_grad += 1
    return gradient
end

function LSQgrad(nlp :: LSQNLPModel, x :: Vector)
    g = zeros(size(nlp.A,2))
    #nlp.counters.neval_grad += 1 #no need, it calls LSQgrad!
    return LSQgrad!(nlp,x,g)
end

function LSQobjgrad!(nlp :: LSQNLPModel, x :: Vector, g :: Vector)
    res = nlp.A*x-nlp.b
    f=0.5*dot(res,res)
    g[:]= (nlp.A)'*res
    nlp.counters.neval_obj += 1
    nlp.counters.neval_grad += 1
    return f,g
end

function LSQobjgrad(nlp :: LSQNLPModel, x :: Vector)
    #nlp.counters.neval_obj += 1
    #nlp.counters.neval_grad += 1
    #res = nlp.A*x-nlp.b
    #f=0.5*dot(res,res)
    g = zeros(size(nlp.A,2))
    return LSQobjgrad!(nlp, x, g)
end

function LSQhess(nlp :: LSQNLPModel, x :: Vector)
    nlp.counters.neval_hess += 1
    return H = nlp.A' * nlp.A
end

function LSQhprod!(nlp :: LSQNLPModel, x:: Vector, v :: Vector, Hv :: Vector)
    nlp.counters.neval_hprod += 1
    Hv[:] = nlp.A' * (nlp.A * v)
    return Hv
end

function LSQhprod(nlp :: LSQNLPModel, x:: Vector, v :: Vector)
    nlp.counters.neval_hprod += 1
    Hv = zeros(size(nlp.A,2))
    return LSQhprod!(nlp,x,Hv)
end
