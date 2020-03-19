##code issus de youtube, voir favoris. rank 1 update avec A_plus = A + u*v' 29:10
##le code renvoie R_plus presentement.
export QR_uv
export Rupdate
export planerot
export MyZupdate

function planerot{T<:Float64}(x::Vector{T}) #https://github.com/JuliaLang/julia/issues/2934
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


function Rupdate{T<:Float64}(Z::Array{T},R::Array{T},u::Vector{T},v::Vector{T})
#u,v tel que A= A+u*v'
m = size(Z,2)
w = Z' * u;
  for i = (m-1):-1:2
    G = planerot(w[i:i+1]);
    w[i:i+1] = G * w[i:i+1];
    R[i:i+1,:] = G * R[i:i+1,:];
  end
Rp = R + w * v';

for i = 1:(m-1)
  G = planerot(Rp[i:i+1,i]);
  Rp[i:i+1,:] = G * Rp[i:i+1,:];
end
return (Rp)
end

function MyZupdate{T<:Float64}(Sₖ::Array{T},Z0::Array{T},R0::Array{T},iterb::Int64,dₖ::Vector{T})
  #Function that calculate the update of R to be able to compute Z
  #by using Sₖ*pinv(R)
  #No need to return Sₖ as the function change it by side effects
  m = size(Sₖ,2)
  println(m)
  if iterb > m
    it       = mod(iterb-1,m)+1
    (u, v)  = QR_uv(Sₖ,dₖ,iterb)
    Rp      = triu(Rupdate(Z0,R0,u,v))
    Sₖ[:,it] = dₖ##########################Side effects
    Zp      = Sₖ * inv(Rp)
  else
    Sₖ[:,iterb] = dₖ
    Zp,Rp   = qr(Sₖ)
  end

  return (Zp,Rp)
end

function MyZupdate{T<:Float64}(Sₖ::Array{T},Z0::Array{T},R0::Array{T},iterb::Int64,dₖ::Vector{T},m::Int64)
  #Function that calculate the update of R to be able to compute Z
  #by using Sₖ*pinv(R)
  #No need to return Sₖ as the function change it by side effects
  if iterb > m
    it       = mod(iterb-1,m)+1
    (u, v)  = QR_uv(Sₖ,dₖ,iterb)
    Rp      = Rupdate(Z0,R0,u,v)
    Sₖ[:,it] = dₖ##########################Side effects
    Zp      = Sₖ * pinv(Rp)
  else
    Sₖ[:,iterb] = dₖ
    Zp,Rp   = qr(Sₖ)
  end

  return (Zp,Rp)
end

##A=ceil(10*rand(5,5))
##[Q,R]=qr(A)
##u=rand(5,1)
##v=rand(5,1)

#code de Alex Townsend: http://www.math.cornell.edu/~web6140/TopTenAlgorithms/QRUpdate.html

function GivensRotation(a::Float64, b::Float64)
    # Calculate the Given's rotation that rotates [a;b] to [r;0]:
c = 0.; s = 0.; r = 0.
if ( b == 0. )
    c = sign(a)
    s = 0.
    r = abs(a)
elseif ( a == 0. )
    c = 0.
    s = -sign(b)
    r = abs(b)
elseif ( abs(a) .> abs(b) )
    t = b/a
    u = sign(a)*abs(sqrt(1+t^2))
    c = 1/u
    s = -c*t
    r = a*u
else
    t = a/b
    u = sign(b)*abs(sqrt(1+t^2))
    s = -1/u
    c = -s*t
    r = b*u
end
    return (c, s, r)
end

function HessenbergQR( R::Matrix )
    # Compute the QR factorization of an upper-Hessenberg matrix:

    n = size(R, 1)
    Q = eye( n )
    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        (c, s, r) = GivensRotation(R[k,k], R[k+1,k])
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] - s*Q[j,k+1]
            Q[j,k+1] = s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end
    return (Q, R)
end

function qrupdate( Q::Matrix, R::Matrix, u::Vector, v::Vector )
    # Compute the QR factorization of Q*R + u*v':

    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = Q'*u
    n = size(Q, 1)

    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        (c, s, r) = GivensRotation(w[k], w[k+1])
        w[k+1] = 0.; w[k] = r
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] - s*Q[j,k+1]
            Q[j,k+1] = s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end
    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] += w[1]*v

    (Q1, R1) = HessenbergQR( R )

    # Return updated QR factorization:
    return (Q*Q1, R1)
end

function HessenbergQR1(R::Matrix )
    # Compute the QR factorization of an upper-Hessenberg matrix:

    n = size(R, 1)

    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        (c, s, r) = GivensRotation(R[k,k], R[k+1,k])
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow

        end
    end
    return (R)
end

function qrupdate1( Q::Matrix, R::Matrix, u::Vector, v::Vector )
    # Compute the QR factorization of Q*R + u*v':

    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = Q'*u


    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        (c, s, r) = GivensRotation(w[k], w[k+1])
        w[k+1] = 0.; w[k] = r
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow

        end
    end
    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] += w[1]*v

    (R1) = HessenbergQR1( R )

    # Return updated QR factorization:
    return (R1)
end

function HessenbergQR2(R::Matrix )
    # Compute the QR factorization of an upper-Hessenberg matrix:

    n = size(R, 1)

    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        (c, s, r) = GivensRotation(R[k,k], R[k+1,k])
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow

        end
    end
    return (R)
end

function qrupdate2( A::Matrix, Q::Matrix, R::Matrix, u::Vector, v::Vector, R1::Matrix )
    # Compute the QR factorization of Q*R + u*v':

    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = Q'*u
    n = size(R, 1)

    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        (c, s, r) = GivensRotation(w[k], w[k+1])
        w[k+1] = 0.; w[k] = r
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow

        end
    end
    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] += w[1]*v

    (R1) = triu(HessenbergQR2( R ))
    R = inv(R1)
    A[:,1] = u
    Q = A * R

    # Return updated QR factorization:
    return (A,Q,R1)
end

function HessenbergQR3(R::Matrix )
    # Compute the QR factorization of an upper-Hessenberg matrix:

    n = size(R, 1)

    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        G,r = givens(R[k,k], R[k+1,k],1,2)
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        R[k:k+1,:] = G * view(R,k:k+1,:)
    end
    return (R)
end

function qrupdate3!( A::Matrix, R::Matrix, d::Vector, i::Int64)
    # Compute the QR factorization of Q*R + u*v' without using Q:
    (u, v) = QR_uv(A,d,i)

    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = (inv(triu(R)))'*(A'*u)
    n = size(R, 1)

    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        G,r = givens(w[k], w[k+1],1,2)
        w[k+1] = 0.; w[k] = r
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        R[k:k+1,:] = G * view(R,k:k+1,:)
    end

    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] += w[1]*v
    (R1) = triu(HessenbergQR3( R ))

    A[:,:] += u*v'
    # Return updated QR factorization:
    return (R1)
end

function QR_uv{T<:Float64}(Sₖ::Array{T},dₖ::Vector{T},iterb::Int64)
##calcul u et v pour faire le Rupdate. i est la colonne a remplacer
    m    = size(Sₖ,2)
    i    = mod(iterb-1,m)+1
    u    = copy(dₖ - Sₖ[:,i]) ##donne la difference a ajouter a notre matrice
    v    = zeros(m);
    v[i] = 1;
    return (u, v)
end
