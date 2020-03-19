using NLPModels
using Optimize

import Compat.view

export LSQNLPModel, LSQobj, LSQgrad, LSQgrad!,
       LSQhess,LSQobjgrad,LSQobjgrad!, LSQhprod, LSQhprod!

function MyLog(x,c)
#c==0 returns the obj evaluation
#c==1 returns the grad
    f = zeros(x)
    g = zeros(x)
    if c == 0
        if any(y->y<=0.0,x)
            f = -Inf * ones(x)
        else
            f = log.(x)
        end
        return f
    else
        if any(y->y<=0.0,x)
            g = -Inf * (abs.(x) + Inf)
        else
            g = 1.0 ./ x
        end#if
        return g
    end
end#function

type LSQNLP_B_Model <: AbstractNLPModel
    A::SparseMatrixCSC
    b::Vector
    # lambda::Real
    # epsi::Real
    meta :: NLPModelMeta
    counters :: Counters
    compteur :: Int64
    rho :: Real
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

function LSQNLP_B_model(A::SparseMatrixCSC, b::Vector;
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

    return LSQNLP_B_Model(A,b,meta,Counters(),0,3.0,Fiability,f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function abc(x,n)
    a = trunc((mod(x-1,n)+1)/n)
    return Int64(a)
end


function LSQobj(nlp :: LSQNLP_B_Model, x :: Vector)
    res = nlp.A*x-nlp.b
    nlp.compteur +=1
    nlp.counters.neval_obj += 1

    if abc(nlp.compteur,30) == 1.0
        nlp.rho /= 1.5
    end

    return 0.5*dot(res,res) - (nlp.rho*sum( MyLog(x,0)))
end

function LSQgrad!(nlp :: LSQNLP_B_Model, x :: Vector, gradient :: Vector)
    res = nlp.A*x-nlp.b
    nlp.compteur +=1
    gradient[:]= (nlp.A)'*res - (nlp.rho * MyLog(x,1))
    nlp.counters.neval_grad += 1
    return gradient
end

function LSQgrad(nlp :: LSQNLP_B_Model, x :: Vector)
    g = zeros(size(nlp.A,2))
    #nlp.counters.neval_grad += 1 #no need, it calls LSQgrad!
    return LSQgrad!(nlp,x,g)
end

function LSQobjgrad!(nlp :: LSQNLP_B_Model, x :: Vector, g :: Vector)
    res = nlp.A * x - nlp.b
    nlp.compteur +=1
    f = 0.5 * dot(res,res)-  (nlp.rho * sum( MyLog(x,0)))
    g[:] = (nlp.A)' * res - (nlp.rho * MyLog(x,1))
    nlp.counters.neval_obj += 1
    nlp.counters.neval_grad += 1
    return f,g
end

function LSQobjgrad(nlp :: LSQNLP_B_Model, x :: Vector)
    #nlp.counters.neval_obj += 1
    #nlp.counters.neval_grad += 1
    #res = nlp.A*x-nlp.b
    #f=0.5*dot(res,res)
    g = zeros(size(nlp.A,2))
    return LSQobjgrad!(nlp, x, g)
end

function LSQhess(nlp :: LSQNLP_B_Model, x :: Vector)
    nlp.counters.neval_hess += 1
    return H = nlp.A' * nlp.A
end

function LSQhprod!(nlp :: LSQNLP_B_Model, x:: Vector, v :: Vector, Hv :: Vector)
    nlp.counters.neval_hprod += 1
    nlp.compteur +=1
    Hv[:] = nlp.A' * (nlp.A * v)
    return Hv
end

function LSQhprod(nlp :: LSQNLP_B_Model, x:: Vector, v :: Vector)
    nlp.counters.neval_hprod += 1
    Hv = zeros(size(nlp.A,2))
    return LSQhprod!(nlp,x,Hv)
end
