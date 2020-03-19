export eval_costpDpR
function eval_costpDpR(x, pD, pR;kwargs...)
  #global k; // globals for graphics of the convergence
  #global obj;
  #global ng;

  X = reshape(x, pR.n, pR.m);
  fGradf = pD.func;
  JGradJ = pR.func;
  vfD,vgD = fGradf(X, pD, pR);
  vfR,vgR = JGradJ(X, pR);

  vf = vfD + pR.lambda * vfR;
  #println(vf[1])
  vg = (vgD + pR.lambda*vgR)[:];

  # Now, let us keep objective and projected gradients norms.
  # projete le gradient et calcul sa norme GUS
  #k=k+1;
  #push!(obj,vf);

  #pg = vg;
  #ilow = find(x.<=pD.Low);
  #if ~isempty(ilow),
  #  pg[ilow] = min(pg(ilow),0);
  #end;
  #iup = find(x.>=pD.Up);
  #if ~isempty(iup),
  #  pg[iup] = max(pg[iup],0);
  #end;
  #push!(ng,norm(pg));

  return vf[1],vg,vfR,vfD,vgR,vgD

end
