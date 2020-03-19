#Brouillon de test

function TestQR(Z,R,Sₖ,dₖ,iterb)
  m = size(Sₖ,2)
  i    = mod(iterb-1,m)+1
  return norm(Sₖ[:,i]-dₖ) < 0.000001
end#function

function TestZR(Z,R,Sₖ,iterb,m;normee::Bool=false)
    i    = mod(iterb-1,m)+1
    println("size(Z)=$(size(Z)), size(R)=$(size(R)), size(Sₖ)=$(size(Sₖ))")
  if normee == false
    return norm(Sₖ[:,1:i] - (Z*R)[:,1:i],2) < 0.000001
  else
    return norm(Sₖ[:,1:i] - (Z*R)[:,1:i],2) < 0.000001, norm(Sₖ[:,1:i] - (Z*R)[:,1:i],2)
  end#if
end#function

function TESTHZCG(;n::Int=20)
include("HZCG6_0_1.jl")
include("arwheadJo.jl")
prob1 = MathProgNLPModel(arwheadJo(n));
HZCG(prob1)
end#function

function TestMyZ(n,m,i)
  S=rand(n,m)
  Q,R=qr(S)
  iter=1
  while iter < i
    d=rand(n)
    (Q,R) = MyZupdate(S,Q,R,iter,d)
    iter +=1
    testtt = norm(Q*R - S)
    println("$iter, $testtt")
  end
end#function

function TESTHZCG(;kwargs...)

# include("HZCG6_0_6.jl");
# include("arwheadJo.jl");
# include("brownden.jl");
# include("hs1.jl");
# include("palmer1c.jl");
# include("srosenbr.jl");
# include("woods.jl");
# include("fminsrf2.jl");
# include("scosine.jl");

prob1 = MathProgNLPModel(arwheadJo(;n=500, kwargs...));
prob2 = MathProgNLPModel(brownden());
prob3 = MathProgNLPModel(hs1());
prob4 = MathProgNLPModel(palmer1c());
prob5 = MathProgNLPModel(srosenbr());
prob6 = MathProgNLPModel(woods());
prob7 = MathProgNLPModel(fminsrf2());
prob8 = MathProgNLPModel(scosine());
(x1, f1, LolBravo, iter1, optimal1, tired1, status1) = HZCG6(prob1;normal=true)
(x2, f2, LolBravo, iter2, optimal2, tired2, status2) = HZCG6(prob2;normal=true)
(x3, f3, LolBravo, iter3, optimal3, tired3, status3) = HZCG6(prob3;normal=true)
(x4, f4, LolBravo, iter4, optimal4, tired4, status4) = HZCG6(prob4;normal=true)
(x5, f5, LolBravo, iter5, optimal5, tired5, status5) = HZCG6(prob5;normal=true)
(x6, f6, LolBravo, iter6, optimal6, tired6, status6) = HZCG6(prob6;normal=true)
(x7, f7, LolBravo, iter7, optimal7, tired7, status7) = HZCG6(prob7;normal=true)
(x8, f8, LolBravo, iter8, optimal8, tired8, status8) = HZCG6(prob8;normal=true)
return status1,status2,status3,status4,status5,status6,status7,status8
end
