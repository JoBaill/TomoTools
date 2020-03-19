using ToeplitzMatrices
using PyPlot

include("TypeTomo.jl")
include("ToeplitzMat.jl")
include("LtransM.jl")
include("eval_costpDpR.jl")
include("func_pRpD.jl")
include("denoise_bound_initJPD.jl")
include("Lforward.jl")
include("project.jl")

function Tomo_tv_fista_A_b2(pD,
                            pR;
                            Affichage :: Bool=false,
                            mon :: Bool=false,
                            epsilon :: Float64=1e-4,
                            MAXITER :: Int=100,
                            denoiseiter :: Int=10,
                            tv :: String="iso",
                            BC :: String="reflexive",#pas encore utile,
                            kwargs...)
#[FistaX,fun_all,ng_all]=TOMO_tv_fista_A_b2(pD.b,pD.A,pR.lambda,pD.Low,pD.Up,pars,L_cost)

  fun_all=[];
  ng_all=[];

  n,m = size(pD.A);
  x_iter = zeros(m);
  y=x_iter;
  t_new=1;

  P1 = [];
  P2 = [];

  L,l,u = pD.L,pD.low,pD.up
  lambda,epsil  = pR.lambda,pR.epsil
##Test
x_old = Inf
grad=zeros(m)

for i=0:MAXITER

  x_old=x_iter;
  t_old=t_new;

grad = pD.A'*(pD.A*y - pD.b);#TODO grad(LSQ,y)

  y=y-(1/L)*grad;

  Y=reshape(y,pR.n,pR.m);
(Z_iter,iter,fun_denoise,P1,P2)=fgp_denoise_bound_init(Y,lambda/L,l,u,P1,P2;
                                      M1=pR.M1,M2=pR.M2,kwargs...)#epsilon,tv,MAXITER,
  z_iter = Z_iter[:];

vf,vg,vfR,vfD,vgR,vgD = eval_costpDpR(z_iter,pD,pR;kwargs...)#TODO objgrad(LASSO,z_iter)


    if mon==0
      x_iter=z_iter;
    else
      if i>1
	      fun_val_old=fun_all[end];
	      if vf>fun_val_old
            x_iter=x_old;
            vf=fun_val_old;
	      else
            x_iter=z_iter;
	      end #if
      end #if
    end #if

    push!(fun_all,vf);

    push!(ng_all,norm(vg));

    t_new=(1+sqrt(1+4*t_old^2))/2;
    y=x_iter+(t_old/t_new)*(z_iter-x_iter)+((t_old-1)/t_new)*(x_iter-x_old);

    i=i+1

end #for
  if Affichage
    X=reshape(x_iter,128,128)
    figure(1)
    imshow(X,cmap=(ColorMap("gray")))
  end


  return x_iter,fun_all,ng_all
end #function
