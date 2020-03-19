using JLD
using PyPlot
using Distributions
using Stopping
using JuMP,Optimize
using NLPModels

#Parametre de feu:
#b=2000*b
#lambda=0.005
#i va jusqu'a 60

plt=PyPlot;

include("../TV.jl")
include("../LLs.jl")
include("../LL.jl")


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
lambda=0.005;

X=0.5*ones(m,n);
TVP=MathProgNLPModel(TV(X,lambda;espi=0.001));
LLS = MathProgNLPModel(LLs(b,b));
LLP=LLmodel(A,LLS, TVP);

atol = 1e-5
rtol = 1e-8

verbose = true
using Lbfgsb
#for i=0:5

stop = TStopping(atol = atol, rtol = rtol, max_iter = 5, max_eval = 1000, max_time = 1000.0)
(x1, f1, gNorm1, iter1, optimal1, tired1, status1)=LbfgsBS(LLP;verbose=verbose, stp=stop)
F1 = LLP.fgraph[2:end]
G1 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 2")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 10, max_eval = 1000, max_time = 1000.0)
(x2, f2, gNorm2, iter2, optimal2, tired2, status2)=LbfgsBS(LLP;verbose=verbose, stp=stop)
F2 = LLP.fgraph[2:end]
G2 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 3")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 25, max_eval = 1000, max_time = 1000.0)
(x3, f3, gNorm3, iter3, optimal3, tired3, status3)=LbfgsBS(LLP;verbose=verbose, stp=stop)
F3 = LLP.fgraph[2:end]
G3 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 4")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 50, max_eval = 1000, max_time = 1000.0)
(x4, f4, gNorm4, iter4, optimal4, tired4, status4)=LbfgsBS(LLP;verbose=verbose, stp=stop)
F4 = LLP.fgraph[2:end]
G4 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

x1=reshape(x1,128,128)
x2=reshape(x2,128,128)
x3=reshape(x3,128,128)
x4=reshape(x4,128,128)

figure(21)

x1=max.(0,x1)
x2=max.(0,x2)
x3=max.(0,x3)
x4=max.(0,x4)

x1=min.(1500.0,x1)
x2=min.(1500.0,x2)
x3=min.(1500.0,x3)
x4=min.(1500.0,x4)

subplot(2,2,1);imshow(x1,cmap=(ColorMap("gray")));
title("LBFGS_10_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,2);imshow(x2,cmap=(ColorMap("gray")));
title("LBFGS_20_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,3);imshow(x3,cmap=(ColorMap("gray")));
title("LBFGS_50_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,4);imshow(x4,cmap=(ColorMap("gray")));
title("LBFGS_100_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);


using LSDescentMethods

stop = TStopping(atol = atol, rtol = rtol, max_iter = 5, max_eval = 1000, max_time = 1000.0)
(X1, f1, gNorm1, iter1, optimal1, tired1, status1)=NewlbfgsS(LLP;verbose=verbose, stp=stop)
FFF1 = LLP.fgraph[2:end]
G1 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 2")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 10, max_eval = 1000, max_time = 1000.0)
(X2, f2, gNorm2, iter2, optimal2, tired2, status2)=NewlbfgsS(LLP;verbose=verbose, stp=stop)
FFF2 = LLP.fgraph[2:end]
G2 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 3")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 25, max_eval = 1000, max_time = 1000.0)
(X3, f3, gNorm3, iter3, optimal3, tired3, status3)=NewlbfgsS(LLP;verbose=verbose, stp=stop)
FFF3 = LLP.fgraph[2:end]
G3 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

println("LBFGS without bounds 4")
stop = TStopping(atol = atol, rtol = rtol, max_iter = 50, max_eval = 1000, max_time = 1000.0)
(X4, f4, gNorm4, iter4, optimal4, tired4, status4)=NewlbfgsS(LLP;verbose=verbose, stp=stop)
FFF4 = LLP.fgraph[2:end]
G4 = LLP.Ggraph[2:end]
reset!(LLP); LLP.fgraph = [0.0]
LLP.Ggraph = [0.0]

X1=reshape(X1,128,128)
X2=reshape(X2,128,128)
X3=reshape(X3,128,128)
X4=reshape(X4,128,128)

X1=max.(0,X1)
X2=max.(0,X2)
X3=max.(0,X3)
X4=max.(0,X4)

X1=min.(1500.0,X1)
X2=min.(1500.0,X2)
X3=min.(1500.0,X3)
X4=min.(1500.0,X4)

figure(22)

subplot(2,2,1);imshow(X1,cmap=(ColorMap("gray")));
title("newLBFGS_10_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,2);imshow(X2,cmap=(ColorMap("gray")));
title("newLBFGS_20_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,3);imshow(X3,cmap=(ColorMap("gray")));
title("newLBFGS_50_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

subplot(2,2,4);imshow(X4,cmap=(ColorMap("gray")));
title("newLBFGS_100_function")
ax=plt.gca()
ax[:axes][:get_xaxis]()[:set_ticks]([]);
ax[:axes][:get_yaxis]()[:set_ticks]([]);

figure(23)

subplot(2,2,1);
suptitle("Evolution of the objective function per iteration")

F1=F1[find(!isnan(F1))]
deleteat!(F1,7)
FFF1=FFF1[find(!isnan(FFF1))]
plot(F1[2:5],label="LBFGS")
plot(FFF1[2:5],label="NewLBFGS")
legend(loc="upper right")
#yscale("log");

F2=F2[find(!isnan(F2))]
deleteat!(F2,7)
FFF2=FFF2[find(!isnan(FFF2))]
subplot(2,2,2);
plot(F2[2:10],label="LBFGS")
plot(FFF2[2:10],label="NewLBFGS")
legend(loc="upper right")
#yscale("log");

F3=F3[find(!isnan(F3))]
deleteat!(F3,7)
FFF3=FFF3[find(!isnan(FFF3))]
subplot(2,2,3);
plot(F3[2:25],label="LBFGS")
plot(FFF3[2:25],label="NewLBFGS")
legend(loc="upper right")
#yscale("log");

F4=F4[find(!isnan(F4))]
deleteat!(F4,7)
FFF4=FFF4[find(!isnan(FFF4))]
subplot(2,2,4);
plot(F4[2:50],label="LBFGS")
plot(FFF4[2:50],label="NewLBFGS")
legend(loc="upper right")
#yscale("log");




#
# println("LBFGS without bounds 4")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 50, max_eval = 1000, max_time = 1000.0)
# (X4, f4, gNorm4, iter4, optimal4, tired4, status4)=NewlbfgsS(LLP;verbose=verbose, stp=stop)
# FF4 = LLP.fgraph[2:end]
# G4 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 5")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 5+10*i, max_eval = 1000, max_time = 1000.0)
# (x5, f5, gNorm5, iter5, optimal5, tired5, status5)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F5 = LLP.fgraph[2:end]
# G5 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 6")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 6+10*i, max_eval = 1000, max_time = 1000.0)
# (x6, f6, gNorm6, iter6, optimal6, tired6, status6)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F6 = LLP.fgraph[2:end]
# G6 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 7")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 7+10*i, max_eval = 1000, max_time = 1000.0)
# (x7, f7, gNorm7, iter7, optimal7, tired7, status7)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F7 = LLP.fgraph[2:end]
# G7 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 8")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 8+10*i, max_eval = 1000, max_time = 1000.0)
# (x8, f8, gNorm8, iter8, optimal8, tired8, status8)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F8 = LLP.fgraph[2:end]
# G8 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 9")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 9+10*i, max_eval = 1000, max_time = 1000.0)
# (x9, f9, gNorm9, iter9, optimal9, tired9, status9)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F9 = LLP.fgraph[2:end]
# G9 = LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
#
# println("LBFGS without bounds 10")
# stop = TStopping(atol = atol, rtol = rtol, max_iter = 10+10*i, max_eval = 1000, max_time = 1000.0)
# (x10, f10, gNorm10, iter10, optimal10, tired10, status4)=LbfgsBS(LLP;verbose=verbose, stp=stop)
# F10 = LLP.fgraph[2:end]
# G10= LLP.Ggraph[2:end]
# reset!(LLP); LLP.fgraph = [0.0]
# LLP.Ggraph = [0.0]
# figure(101)
# x1=reshape(x1,128,128)
# x2=reshape(x2,128,128)
# x3=reshape(x3,128,128)
# x4=reshape(x4,128,128)
#
# x1=max.(0,x1)
# x2=max.(0,x2)
# x3=max.(0,x3)
# x4=max.(0,x4)
# # x5=reshape(x5,128,128)
# # x6=reshape(x6,128,128)
# # x7=reshape(x7,128,128)
# # x8=reshape(x8,128,128)
# # x9=reshape(x9,128,128)
# # x10=reshape(x10,128,128)
#
# # a=1+i*10
# # b=10+i*10
# #subplot(2,5,1);imshow(x1,cmap=(ColorMap("gray")));
# subplot(2,2,1);imshow(x1,cmap=(ColorMap("gray")));
# # suptitle("graph $a a $b")
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
# subplot(2,2,2);imshow(x2,cmap=(ColorMap("gray")));
# #subplot(2,5,2);imshow(x2,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
# subplot(2,2,3);imshow(x3,cmap=(ColorMap("gray")));
# #subplot(2,5,3);imshow(x3,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
# subplot(2,2,4);imshow(x4,cmap=(ColorMap("gray")));
# #subplot(2,5,4);imshow(x4,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);

# subplot(2,5,5);imshow(x5,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
#
# subplot(2,5,6);imshow(x6,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
#
# subplot(2,5,7);imshow(x7,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
#
# subplot(2,5,8);imshow(x8,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
#
# subplot(2,5,9);imshow(x9,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);
#
# subplot(2,5,10);imshow(x10,cmap=(ColorMap("gray")));
# title("LBFGS")
# ax=plt.gca()
# ax[:axes][:get_xaxis]()[:set_ticks]([]);
# ax[:axes][:get_yaxis]()[:set_ticks]([]);

#end
#x1=max(0,x1) enleve le mauvais contraste
