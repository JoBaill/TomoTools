using ToeplitzMatrices
export ToeplitzMat


function ToeplitzMat(X)
  nr,nc = size(X)

  r2 = zeros(1,nc);   r2[1] = 1;
  c2 = zeros(1,nc);   c2[1] = 1;   c2[2] = -1;
  M2 = full(Toeplitz(vec(c2), vec(r2)))
  M2[nc,nc] = 0

  r1 = zeros(1,nr);   r1[1] = 1;
  c1 = zeros(1,nr);   c1[1] = 1;   c1[2] = -1;
  M1 = full(Toeplitz(vec(r1), vec(c1)))
  M1[nr,nr] = 0

  M1 = sparse(M1);
  M2 = sparse(M2);

  return M1,M2
end
