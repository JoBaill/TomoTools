TEST QR UPDATE TOWNSEND MODIFIÉ

A=ceil(10*rand(7,4))
QQ,RR = qr(A);
d=1.0*[1,2,3,4,5,6,7]
u,v=QR_uv(A,d,1)
#U*V'+A
#A
RRR = qrupdate3(QQ,RR,u,v)

RRRR = inv(RRR)

AA = copy(A)
AA[:,1] = d

norm(AA * RRRR - qr(AA)[1])


#TEST CARRÉ TOWNSEND

A=  [2.0  1.0  7.0  5.0
    10.0  6.0  3.0  9.0
    7.0  8.0  4.0  3.0
    1.0  7.0  7.0  9.0]

q,r = qr(A)
d=1.0*[1,2,3,4]
u,v=QR_uv(A,d,1)
A+u*v'

QQQ,RRR = qrupdate(q,r,u,v)

QQQ*RRR

#TEST CARRÉ Jo

A=  [2.0  1.0  7.0  5.0
    10.0  6.0  3.0  9.0
    7.0  8.0  4.0  3.0
    1.0  7.0  7.0  9.0]

q,r = qr(A)
d=1.0*[1,2,3,4]
u,v=QR_uv(A,d,1)
AA = A+u*v'

QQQ,RRR = qrupdate3(A,q,r,u,v)


RRRR = inv(RRR)
Qfinal = AA * RRRR

#MULTITEST CARRÉ Jo
function g(n,m)
G,o = givens(0.1,0.2,1,2)
#création des matrices
A=ceil.(10*rand(n,m))
AAA=copy(A)
AA=copy(A)

q,r = qr(A);
d=Array(vec(1.0*1:n))
u,v=QR_uv(A,d,1)
#U*V'+A
#A
@time r=qrupdate3!(A,q,r,u,v)

qqq,rrr = qr(AAA)
d=Array(vec(1.0*1:n))
u,v=QR_uv(AAA,d,1)

@time qqq,rrr = MyZupdate(AAA,qqq,rrr,1,d)

qq,rr = qr(AA);
d=Array(vec(1.0*1:n))
u,v=QR_uv(AA,d,1)
#U*V'+A
#A
@time qq,rr=qrupdate(qq,rr,u,v)
end


#MULTITEST RECTANGLE Jo
function g(n,m)
G,o = givens(0.1,0.2,1,2)
#création des matrices
A=ceil.(10*rand(n,m))
AAA=copy(A)
AA=copy(A)

q,r = qr(A);
d=Array(vec(1.0*1:n))
u,v=QR_uv(A,d,1)
#U*V'+A
#A
@time r=qrupdate3!(A,q,r,u,v,G)

qqq,rrr = qr(AAA);
d=Array(vec(1.0*1:n));
u,v=QR_uv(AAA,d,1);

@time qqq,rrr = MyZupdate(AAA,qqq,rrr,12,d)
end


#TEST ANCIEN RECTANGLE Jo
function h(n)
A=ceil(10*rand(7,4))
d=1.0*[1,2,3,4,5,6,7]
q,r = qr(A)
@time q,r = MyZupdate(A,q,r,1,d)
