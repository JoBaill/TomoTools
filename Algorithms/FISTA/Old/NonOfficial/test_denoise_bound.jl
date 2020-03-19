function test_denoise_bound()

P2=[3.0 3.0 3.0;
    2.0 0.0 0.0;
    2.0 4.0 4.0;
    0.0 0.0 0.0]

(Aa,bb)=fgp_denoise_bound_init(P2,
                               4,
                               -Inf,
                               Inf;
                               epsilon = 1e-4,
                               MAXITER = 100,
                               tv = "l1")

Real = [1.75083 1.75083 1.75083;
        1.75034 1.75034 1.75034;
        1.74966 1.74966 1.74966;
        1.74917 1.74917 1.74917]

test = sum(Aa-Real) < 0.01
testIter = (bb <= 77)

if bb < 77
  println("meilleur algo qu'avant!")
end
#println(sum(Aa-Real))
return(test&testIter)
end
