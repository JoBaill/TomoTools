using NLPModels
using Optimize

import Compat.view

export LogLModel, obj, grad, grad!,
       hess,objgrad,objgrad!,hprod,hprod!

import NLPModels.obj
import NLPModels.grad
import NLPModels.grad!
import NLPModels.hess
import NLPModels.objgrad
import NLPModels.objgrad!
import NLPModels.hprod
import NLPModels.hprod!


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

type LogLModel <: AbstractNLPModel
    A::SparseMatrixCSC
    b::Vector
    i ::Array{Float64,1}
    j ::Array{Float64,1}
    meta :: NLPModelMeta
    counters :: Counters
    compteur :: Int64 #pour faire diminuer la pénalité Log
    rho :: Real
    fgraph :: Array{Float64,1}
    Ggraph :: Array{Float64,1}
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

function LogLmodel(A::SparseMatrixCSC, b::Vector;
    Fiability :: Function = fiability,
    i::Array{Float64,1} = zeros(size(A,2)),
    j::Array{Float64,1} = zeros(size(A,2)),
    f::Function = obj,
    g::Function = grad,
    g!::Function = grad!,
    fg::Function = objgrad,
    fg!::Function = objgrad!,
    #H::Function = LASSOhess,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = NotImplemented,#hprod,
    Hp!::Function = NotImplemented,#hprod!
    )

    fgraph = [0.0]#initialize graph tools
    Ggraph = [0.0]

    m = size(A,2)
    srand(1234);
    #Xₖ = ((1.0 / (m) ) * sum(b)) * ones(m) + rand(m)
    Xₖ = copy(A' * b);
    X = reshape(Xₖ, Int64(sqrt(length(Xₖ))), Int64(sqrt(length(Xₖ))))

    meta = NLPModelMeta(m;x0=Xₖ,lvar=zeros(m))
    println("minimum of X = $(minimum(X))")
    return LogLModel(A,b,i,j,meta,Counters(),0,5.0,fgraph,Ggraph,Fiability,
                        f,g,g!,fg,fg!,H,Hcoord,Hp,Hp!)
end#function

function obj(nlp :: LogLModel, x :: Vector)
    # println("size(A)=$(size(A))")
    # println("size(x)=$(size(x))")
    Ax = nlp.A * x
    nlp.compteur +=1
    f = sum(Ax - nlp.b .* log.(Ax))
    nlp.counters.neval_obj += 1
    push!(nlp.fgraph,f)
    return f
end

function grad!(nlp :: LogLModel, x :: Vector, gradient :: Vector)
    Ax = nlp.A * x
    nlp.compteur +=1

    gradient[:] = nlp.A' * ( ones(nlp.b) - (nlp.b .* (1 ./ Ax)) )
    #gradient[:] = nlp.A' * ( ones(nlp.b) - (nlp.b ./ (j + eps() ) ) )
    nlp.counters.neval_obj += 1
    push!(nlp.Ggraph,norm(gradient))
    return gradient
end

function grad(nlp :: LogLModel, x :: Vector)
    g = zeros(nlp.meta.nvar)
    #nlp.counters.neval_grad += 1 #pas de besoin car on call grad!
    return grad!(nlp,x,g)
end

function objgrad!(nlp :: LogLModel, x :: Vector, gradient :: Vector)
    # nlp.counters.neval_obj += 1 #pas besoin, on call obj et grad
    # nlp.counters.neval_grad += 1
    Ax = nlp.A * x
    nlp.compteur +=1
    #println(nlp.rho)

    f = sum(Ax - nlp.b .* log.(Ax))
    gradient[:] = nlp.A' * ( ones(nlp.b)- (nlp.b .* (1 ./ Ax)))
    return f, gradient
end

function objgrad(nlp :: LogLModel, x :: Vector)
    # nlp.counters.neval_obj += 1 #pas besoin, on call obj et grad
    # nlp.counters.neval_grad += 1
    g = zeros(nlp.meta.nvar)
    return objgrad!(nlp,x,g)
end
