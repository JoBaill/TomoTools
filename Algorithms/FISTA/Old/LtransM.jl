function LtransM(X,M1,M2)

    #[m,n]=size(X);

    #P1=X(1:m-1,:)-X(2:m,:);
    #P2=X(:,1:n-1)-X(:,2:n);

    P1t = M1 * X;
    P1 = P1t[1:(end-1), :];

    P2t = X * M2;
    P2 = P2t[:, 1:(end-1)];

    return P1,P2
end
