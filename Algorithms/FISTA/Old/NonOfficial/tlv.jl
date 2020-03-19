function tlv(X;
             tv:: String= "iso")

m,n=size(X);
P3,P4=Ltrans(X);


if tv == "iso"
  D=zeros(m,n);
  D[1:m-1,:]=P3.^2;
  D[:,1:n-1]=D[:,1:n-1]+P4.^2;
  out=sum(sqrt(D));
elseif tv == "l1"
  out=sum(sum(abs(P3)))+sum(sum(abs(P4)));
else
  error("Invalid total variation type. Should be either 'iso' or 'l1'");
end

return out
end
