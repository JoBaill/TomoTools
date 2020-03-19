function Ltrans(X)

m,n=size(X);

P1=X[1:m-1,:]-X[2:m,:];
#donne le gradient en y d'une image

P2=X[:,1:n-1]-X[:,2:n];
#donne le gradient en x d'une image

return P1,P2

end
