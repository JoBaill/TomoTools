using NLPModels
using Optimize

import Compat.view

import NLPModels.obj
import NLPModels.grad
import NLPModels.grad!

export SNLPModel, obj, grad, grad!

type SNLPModel <: AbstractNLPModel
  NLP :: AbstractNLPModel
  X::Vector
  Z::Array
  L_gx::Vector
  L_gα::Vector
  meta :: NLPModelMeta
  counters :: Counters


  # Functions
  f :: Function
  g :: Function
  g! :: Function

end#Type

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

function SNLPmodel(NLP, X, Z;
    f::Function = obj,
    g::Function = grad,
    g!::Function = grad!
    )
  L_gx = Array{Float64}(X)
  nvar = size(Z,2)
  L_gα = Array{Float64}(nvar)
  meta = NLPModelMeta(nvar,x0=zeros(nvar))

  return SNLPModel(NLP,copy(X),Z,L_gx,L_gα,meta,Counters(),f,g,g!)
end#function

function obj(nlp :: SNLPModel, α :: Vector)
  f=obj(nlp.NLP,nlp.X + nlp.Z * α)
  #nlp.counters.neval_obj += 1
  return f
end

function grad(nlp :: SNLPModel, α :: Vector)
  nlp.L_gα[:] = grad!(nlp, α, nlp.L_gα)
  #nlp.counters.neval_grad += 1
  return nlp.L_gα
end

function grad!(nlp :: SNLPModel, α :: Vector, gradient :: Vector)
  Xα= nlp.X + nlp.Z * α
  nlp.L_gx[:] = grad(nlp.NLP, Xα)
  nlp.L_gα[:] = (nlp.Z)' * nlp.L_gx
  gradient[:] = copy(nlp.L_gα)
  #nlp.counters.neval_grad += 1
  return gradient
end
