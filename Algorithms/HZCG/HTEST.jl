include("HZCG6_0_6_1.jl");
include("arwheadJo.jl");
prob1 = MathProgNLPModel(arwheadJo(;n=500));
HZCG6(prob1)
