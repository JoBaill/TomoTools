function deblur_tv_Fista()

  # initialization
  X_iter=Bobs;
  Y=X_iter;
  t_new=1;

  for i=1:MAXITER
    # store the old value of the iterate and the t-constant
    X_old=X_iter;
    t_old=t_new;
    # gradient step
    #D=Sbig.*trans(Y)-Btrans;
    #Y=Y-2/L*itrans(conj(Sbig).*D);
    #Y=real(Y);
    




end
