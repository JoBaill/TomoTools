test_Tomo_tv_fista_A_b2
m=10
n=10
XX=zeros(m,n);
M1,M2 = ToeplitzMat(XX);


A=eye(n*m,n*m)
b=rand(n*m,1)

#A=rand(2*m*n,m*n);
#b=rand(2*m*n,1);

L  = 2*eig(A'*A)[1][end]

pD=Data_terms(copy(b),n,m,To,copy(A),0,1,copy(L));
pR=Regul_terms(0.005,0.0001,10,10,RTVs,copy(M1),copy(M2));
