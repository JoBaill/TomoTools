#####Création de Phantoms bruités#####
## On veut créer les phantoms bruités et les matrices systèmes sans lignes de 0
## associés. Va créer des versions modifiés de Asmall et ABig pour chaque
## sinogram.
## Les matrice seront appelé avec: load("AsmallDeathstar128_20k.jld", "AsmallDeathstar128_20k")
## Il y en aura une différente pour les différentes matrice système (Asmall et ABIG),
## pour les différentes image (Deathstar et Shepp_Logan), pour les différentes
## tailles d'image (128 et 256(à venir)) et pour les différents nombre de compte
## (20k,100k,500k), ce qui ferait 2.4 Gig... Donc nous n'allons garder que les
## sinogram non classé, et feront l'ablation des zéros à chaque utilisation à la
## place.
##
## On a accès au sinogram avec la commande:
##
## load("noisyDeathstar128Asmall_20k.jld","b")
##
## Il faudra donc utiliser les commandes:
##            N1=find(nonzero,b1)
##            A11=sparse(A11[N1,:])
##            b1=b1[N1]

using JLD
using PyPlot
using Distributions

#TODO: I = I₀*(E.^(-A2X)), and set Ax = b := -ln(I/I₀)
#I₀ = 100,1000 etc.
#A2X=A*Image
#il faut "Poisonner" I
#puis créer b

A1 = load("../../MatriceCTAdam/Apetit.jld","Apetit");
A2 = load("../../MatriceCTAdam/Agrand.jld","Agrand");
#A3 = load("../../MatriceCTAdam/ABBBIGPET.jld","A")
println(size(A1))
println(size(A2))

# Image1 = load("../Mémoire_Ressources/Shepp_Logan_y128_20k.jld","y128_20k");
# Image2 = load("../Mémoire_Ressources/Shepp_Logan_y256_20k.jld","y256_20k");
# Image3 = load("../Mémoire_Ressources/Shepp_Logan_y128_100k.jld","y128_100k");
# Image4 = load("../Mémoire_Ressources/Shepp_Logan_y256_100k.jld","y256_100k");
# Image5 = load("../Mémoire_Ressources/Shepp_Logan_y128_500k.jld","y128_500k");
# Image6 = load("../Mémoire_Ressources/Shepp_Logan_y256_500k.jld","y256_500k");
#
# Image7 = load("../Mémoire_Ressources/Deathstar_x128_20k.jld","x128_20k");
# Image8 = load("../Mémoire_Ressources/Deathstar_x256_20k.jld","x256_20k");
# Image9 = load("../Mémoire_Ressources/Deathstar_x128_100k.jld","x128_100k");
# Image10 = load("../Mémoire_Ressources/Deathstar_x256_100k.jld","x256_100k");
# Image11 = load("../Mémoire_Ressources/Deathstar_x128_500k.jld","x128_500k");
# Image12 = load("../Mémoire_Ressources/Deathstar_x256_500k.jld","x256_500k");

# b = copy(A*Image[:])
# Distribution = Poisson.(b);
# b = rand.(Distribution);

function nonzero(x)
    return abs(x) > eps()
end

names = ["CTDeathstar", "CTShepp_Logan"]
#num = [128]#,256] #256
num = [128]

#num = [256]
variables = ["x","y"]
#counts = ["20k","100k","500k"]
counts = ["100","1000"]
# SystemMatrix = ["Asmall","ABIG","ABBBIGPET"] DSA
SystemMatrix = ["Apetit","Agrand"]
# AA = [A1,A2,A3] DSA
AA = [A1, A2]
compte = [100,1000]

X = [];
Y = [];
#Z = [];

A11 = [];
A22 = [];
#A33 = [];

b1 = [];
b2 = [];
#b3 = [];

N1 = [];
N2 = [];
#N3 = [];

Distribution1 = [];
Distribution2 = [];
#Distribution3 = [];



for j in num #pour les images de 128 et 256 pixel

    for k in 1:length(counts) # pour 20, 100 et 500k counts

            X = load(string("../../Counted_PhantomCT/$(names[1])_$(variables[1])$(j)_$(counts[k]).jld"), string("$(variables[1])$(j)_$(counts[k])"))

            Y = load(string("../../Counted_PhantomCT/$(names[2])_$(variables[2])$(j)_$(counts[k]).jld"), string("$(variables[2])$(j)_$(counts[k])"))

        for A in 1:2#3#1:2
            b1 = compte[k] * ((e*ones(size(AA[A],1))) .^ (-1 * copy(AA[A]*X[:]))) #A1*Image1, puis A2 * Image1
            Distribution1 = Poisson.(b1); #On crée des lois de Poisson
            b1 = rand.(Distribution1); #On échantillonne selon nos lois de Poisson
            b1 = -log.((b1+0.01) / compte[k])#Comme ça on crée les ressources pour Ax=b
            b1[b1.< 0.0]=0.0;#on patch car Ax est toujours non-négatif

            b2 = compte[k] * ((e*ones(size(AA[A],1))) .^ (-1 * copy(AA[A]*Y[:]))) #100*e.^A1*Image2 et Image1, puis 1000*A1 * Image2 et Image1
            Distribution2 = Poisson.(b2); #On crée des lois de Poisson
            b2 = rand.(Distribution2); #On échantillonne selon nos lois de Poisson
            b2 = -log.((b2+0.01) / compte[k])#Comme ça on crée les ressources pour Ax=b
            b2[b2.< 0.0]=0.0;#on patch car Ax est toujours non-négatif

            save(string("noisyLS$(names[1])$(j)$(SystemMatrix[A])_$(counts[k]).jld"), "b", b1)#noisyDeathstarAsmall128_20k.jld
            save(string("noisyLS$(names[2])$(j)$(SystemMatrix[A])_$(counts[k]).jld"), "b", b2)
        end
    end
end
