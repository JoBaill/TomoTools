include("func_pRpD.jl")
include("ToeplitzMat.jl")


type Data_terms
    b       #the signal
    n::Int  #Dimensions of A
    m::Int  #
    func :: Function   #objective function
    A       #system Matrix
    low     #lower bound
    up      #upper bound
    L       #Guess of the Lipschitz constant of the gradient of ||A(X)-Bobs||^2
end

type Regul_terms
    lambda    #
    epsil   #smoothing parameter
    n :: Int  #Size of the image
    m :: Int  #
    func :: Function    #regul function
    M1        #Toeplitz matrix trick
    M2        #Toeplitzmat(image)
end

function Regul(lambda, n, m; func::Function=RTVs, epsil::Float64=0.0)
  M1,M2 = ToeplitzMat(eye(n, m))
  return Regul_terms(lambda, epsil, n, m, func, M1, M2)
end


function data(A, b, func, L; low::Float64=0.0, up::Float64=sum(b))
  n,m = size(A)
return Data_terms(b, n ,m, To, A, low, up, L)
end
