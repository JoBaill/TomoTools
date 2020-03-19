function MLEM(A::SparseMatrixCSC,
              b::Vector,
              m::Int64,
              n::Int64;
              Affichage :: Bool=false,
              MAXITER :: Int=100,
              epsilon :: Float64 = 1e-4)

length_Xₖ = m*n
Xₖ       = (1/length_Xₖ * sum(b)) * ones(length_Xₖ)
#Xₖₚ       = (1/length_Xₖ * sum(b)) * ones(length_Xₖ)
Norme     = Inf
i=0
  N         = A'*ones(size(b))
  ng=[]
  while i < MAXITER
#    Xₖ        = Xₖₚ

    C         = A'*(b ./ (A*Xₖ))
#    Xₖₚ       = (Xₖ ./ N) .* (C)
    Xₖ       = (Xₖ ./ N) .* (C+eps())
    ∇f        = N-C
    Norme     = norm(∇f)
    push!(ng,Norme)
    i         = i+1
    if mod(i,5)==0
      println(i)
      println("")
   #   X=reshape(Xₖ,128,128);
  #imshow(X)
    end
  end #while
#TODO message d'erreur si on sort avec MAXITER
if i>=MAXITER
  println("MAXITER reached in MLEM")
end
#return Xₖₚ
if Affichage
  X=reshape(Xₖ,128,128);
  imshow(X)
end

return Xₖ,ng

end
