export To, RTVs

function To(X,pD,pR)

ER = pD.A * X[:] - pD.b
f  = 0.5 * ER' * ER
g  = pD.A' * ER
g  = reshape(g, pR.n, pR.m)

return f,g
end

function RTVs(X,pR)

zzz,yyy   = size(X)
Gx        = Array{Float64}(zzz, yyy, 2)
Gx[:,:,1] = pR.M1 * X
Gx[:,:,2] = X * pR.M2
NNs       = sqrt.(pR.epsil^2 + sum(Gx.^2, 3))

Denom     = max(eps(),NNs)
f         = sum(NNs)
GxNormed  = Gx ./ repeat(Denom,outer = [1, 1, 2])

g         = pR.M1' * GxNormed[:, :, 1] + GxNormed[:, :, 2] * pR.M2'

return f,g
end

#pR=Regul(1,500,500)
#@time for i=1:20
#print(i)
#X=rand(500,500);
#RTVs(X,pR);
#end


#@time for i=1:100
#print(i)
#X=rand(250000);
#objgrad(TVP,X);
#end
