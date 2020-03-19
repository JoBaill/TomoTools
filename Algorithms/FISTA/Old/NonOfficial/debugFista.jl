using JLD
using PyPlot
using Distributions
using Stopping
using JuMP,Optimize
using NLPModels


plt=PyPlot;
include("TypeTomo.jl");
include("FISTA_NLP.jl");

include("../TV.jl");
include("../LLs.jl");
include("../LL.jl");


A=load("../ABig.jld","A");
b=load("../bbig.jld","b");

b=2000*b;
Dist = Poisson.(b);
b = rand.(Dist);
function nonzero(x)
return abs(x)>eps()
 end
#
 N=find(nonzero,b)
#
 A=sparse(A[N,:])
b=b[N]
m=128;
n=128;
i=1

#for lambda in [0.4,0.04,0.004,0.0004]
lambda=0.004
figure(i)
X=0.5*ones(m,n);
TVP=MathProgNLPModel(TV(X,lambda;espi=0.001));
LLS = MathProgNLPModel(LLs(A*X[:],b));
LLP=LLmodel(A,LLS, TVP);

pR=Regul(lambda,n,m);

pD=data(A,b,To,10;low=0.0,up=Inf);

x1,fun_all1,ng_all1=Tomo_tv_fista_A_b2(LLP,pD,pR;MAXITER=5)
x1=reshape(x1,128,128)
x2,fun_all2,ng_all2=Tomo_tv_fista_A_b2(LLP,pD,pR;MAXITER=125)
x2=reshape(x2,128,128)
x3,fun_all3,ng_all3=Tomo_tv_fista_A_b2(LLP,pD,pR;MAXITER=50)
x3=reshape(x3,128,128)


subplot(2,2,1);imshow(x1,cmap=(ColorMap("gray")));
title("FISTA_10_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);
subplot(2,2,2);imshow(x2,cmap=(ColorMap("gray")));
title("FISTA_50_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);
subplot(2,2,3);imshow(x3,cmap=(ColorMap("gray")));
title("FISTA_100_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);
i=i+1
#end
