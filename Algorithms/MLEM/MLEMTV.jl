using PyPlot

include("../Metrics.jl")
include("../Mémoire_Ressources/fiability.jl")


function MLEMTV(nlp :: AbstractNLPModel;
                Affichage :: Bool=false,
                stp :: TStopping = TStopping(),
                real_image :: Array{Float64,2} = zeros(2,2),
                ShowAll :: Bool=false,
                TESTING :: Bool=false,
                MAXITER :: Int=1,
                verbose :: Bool=false,
                epsilon :: Float64 = 1e-4)

        MAXITER = stp.max_iter


    PSNR_MLEM = []
    proj_graph = []
    proj = 0
    #β=0.0 #for now DSA Gus

    #β = nlp.lambda #fait la terre si beta = 0.5 avec 100 iter et epsi = 0.001;
    β = 1.0 #notre modele de TV contient deja lambda de maniere multiplicative
    A = nlp.A;
    b = nlp.b;
    n = Int64(sqrt(size(A,2)))
    m = n

    length_Xₖ = m * n;
    srand(1234);
    # Xₖ        = (1 / length_Xₖ * sum(b)) * ones(length_Xₖ) + rand(length_Xₖ)
    # #Xₖₚ       = (1/length_Xₖ * sum(b)) * ones(length_Xₖ)
    Xₖ = nlp.meta.x0
    Norme = Inf
    i     = 0

    N = A' * ones(size(b))
    #println(maximum(N))
    ng = []
    nf = []

    while i < MAXITER
    #    Xₖ        = Xₖₚ
    C  = A' * (b ./ (A * Xₖ))
    #    Xₖₚ       = (Xₖ ./ N) .* (C)
    #Xₖ = (Xₖ ./ (N + β * grad(nlp.TVP, Xₖ[:]))) .* (C + eps())
    ∇f = N - C + grad(nlp.TVP, Xₖ[:])
    #println("gradTV = $(grad(nlp.TVP, Xₖ[:]))")
    Xₖ = (Xₖ ./ (N + β * grad(nlp.TVP, Xₖ[:]))) .* (C)
    nlp.LLs.fiability(Xₖ,nlp.Image,1,nlp.proj_graph,nlp.values) #on met 1 car on fait obj pour le push de nf...

    Norme = norm(∇f,Inf)

    push!(ng,Norme)
    push!(nf, obj(nlp, Xₖ))

    verbose && @printf("%4s  %8s  %11s %8s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "∇f'd", "bk","t","scale","h'(t)","t_original")
    verbose && @printf("%4d  %8e  %7.1e", i, nf[end], Norme)

    i = i + 1

    if mod(i, 100) == 0
        println(i)
        println("")
        if ShowAll
            X = reshape(Xₖ,128,128);
            imshow(X)
        end#if
    end#if

    end #while

    println(Norme)
    return (Xₖ, nf, Norme, i, 0, 0, 0, 0, i, 0)

end
