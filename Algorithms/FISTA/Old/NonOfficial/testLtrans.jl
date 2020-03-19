function testLtrans()

P1=[3.0 3.0 3.0 1.0;
    2.0 1.0 1.0 1.0;
    2.0 4.0 4.0 4.0]

a,b=Ltrans(P1)

aReal=[1.0 2.0 2.0 0.0;
      0.0 -3.0 -3.0 -3.0]
bReal=[0.0 0.0 2.0;
      1.0 0.0 0.0;
      -2.0 0.0 0.0]

return (a==aReal&&b==bReal)

end
