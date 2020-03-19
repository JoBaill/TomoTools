function testLforward()

P1=[3.0 3.0 3.0 1.0;
    2.0 1.0 1.0 1.0;
    2.0 4.0 4.0 4.0]

P2=[3.0 3.0 3.0;
    2.0 0.0 0.0;
    2.0 4.0 4.0;
    0.0 0.0 0.0]

X = Lforward(P1,P2)

Y=[6.0 3.0 3.0 -2.0;
   1.0 -4.0 -2.0 0.0;
   2.0 5.0 3.0 -1.0;
   -2.0 -4.0 -4.0 -4.0]

return (X==Y)

end