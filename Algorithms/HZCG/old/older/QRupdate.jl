##code issus de youtube, voir favoris. rank 1 update avec A_plus = A + u*v' 29:10
##le code renvoie R_plus presentement.
export QR_uv
export Rupdate
export planerot
export MyZupdate

function planerot{T<:Real}(x::Vector{T}) #https://github.com/JuliaLang/julia/issues/2934
    if length(x) != 2
        error()
    end
    da, db = x
    roe = db
    if abs(da) > abs(db)
       roe = da
    end
    scale = abs(da) + abs(db)
    if scale == 0
        c = 1.
        s = 0.
        r = 0.
    else
        r = scale * sqrt((da/scale)^2 + (db/scale)^2)
        r = sign(roe) * r
        c = da/r
        s = db/r
    end
    return [c s; -s c]
end

function QR_uv(Sₖ,dₖ,iterb,m)
##calcul u et v pour faire le Rupdate. i est la colonne a remplacer
  i    = mod(iterb-1,m)+1
  u    = dₖ - Sₖ[:,i] ##donne la difference a ajouter a notre matrice
  v    = zeros(size(Sₖ,2));
  v[i] = 1;
  return (u, v)

end

function Rupdate(Z,R,u,v,m)
#u,v tel que A= A+u*v'
w = Z' * u;
  for i = (m-1):-1:2
    G=planerot(w[i:i+1])
    w[i:i+1]=G*w[i:i+1]
    R[i:i+1,:] = G * R[i:i+1,:];
  end
Rp = R + w * v';

for i = 1:(m-1)
  G = planerot(Rp[i:i+1,i]);
  Rp[i:i+1,:] = G * Rp[i:i+1,:];
end
return (Rp)
end

function MyZupdate(Sₖ,Z0,R0,iterb,dₖ,m)
  #Function that calculate the update of R to be able to compute Z
  #by using Sₖ*pinv(R)
  #No need to return Sₖ as the function change it by side effects
  if iterb > m
    it       = mod(iterb-1,m)+1
    (u, v)  = QR_uv(Sₖ,dₖ,iterb,m)
    Rp      = Rupdate(Z0,R0,u,v,m)
    Sₖ[:,it] = dₖ##########################Side effects
    Zp      = Sₖ * pinv(Rp)
  else
    Sₖ[:,iterb] = dₖ
    Zp,Rp   = qr(Sₖ[:,1:iterb])
  end

  return (Zp,Rp)
end

function TestDirectionSubspace(Z,d,iterb,m)
    for i =1:size(Z,2)
      println(norm(d'*Z[:,i]))
    end#for
end#function

##A=ceil(10*rand(5,5))
##[Q,R]=qr(A)
##u=rand(5,1)
##v=rand(5,1)
