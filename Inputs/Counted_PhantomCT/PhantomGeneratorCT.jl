
using Images
using JLD
using PyPlot
#warn("using ImageView implies that using PyPlot may cause seg fault with figure(::Integer)")

function ndgrid(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end


function phantom(AI,xlow,xhigh,ylow,yhigh,xstep,ystep)

# // PHANTOM generates bit-map Shepp-Logan style
# //  http://hep.ph.liv.ac.uk/~hock/index.html
# //
# //  Code adapté par JPD tiré de Kai Meng Hock
# //
# //  AI contient la description et X le bitmap
# //---------------------------------------
# //
# //  To generate images
# //  using ellipses for testing
# //  tomography reconstruction codes
# //
# //  E.g. input:
# //
# //  Kai Hock
# //  Cockcroft Institute
# //  10 June 2010
# //  adaptation to Scilab by Jean-Pierre Dussault, traducted to Julia by Jonathan Baillargeon
# //---------------------------------------
    x1 = AI[:, 1];   # centre coordinate x
    y1 = AI[:, 2];   # centre coordinate y
    A  = AI[:, 3];   # semi major axis
    B  = AI[:, 4];   # semi minor axis
    a1 = AI[:, 5];   # (deg) rotation angle
    ri = AI[:, 6];   # refractive index
    n1 = length(x1); # number of ellipses

    ##----intensity matrix -------

    x0 = xlow:xstep:xhigh;
    y0 = ylow:ystep:yhigh;
    nx = length(x0);
    ny = length(y0);
    yy, xx = ndgrid(-y0, x0);
    f0 = zeros(ny, nx);

    for i1 = 1:n1
        alpha1 = a1[i1]/180.0*pi;
        x =  (xx-x1[i1])*cos(alpha1) + (yy-y1[i1])*sin(alpha1);
        y = -(xx-x1[i1])*sin(alpha1) + (yy-y1[i1])*cos(alpha1);
        f0 = f0 + ri[i1]*heaviside(1.0 - (x/A[i1]) .^2.0 - (y/B[i1]) .^2.0);
    end
    fmax = maximum(f0);
    f0 = f0 / fmax;

    X = f0[:]; f0[f0 .< 0.0] = 0.0;
    return X
end

function heaviside(x)
    m, n = size(x);
    y = zeros(m, n);
    for i = 1:m
        for j = 1:n
            if x[i, j] > 0.0
                y[i, j] = 1.0;
            elseif x[i, j] == 0.0
                y[i, j] = 0.5;
            end
        end
    end

    return y
end


##-----------------------------------------------------------
##    center          major   minor   rotation   refractive
##    coordinate      axis    axis    angle      index
##     x      y                       (deg)
AI1 = [0.0    0.0     0.92    0.69     90.0        2.0;
       0.0   -0.0184  0.874   0.6624   90.0       -0.98;
       0.22   0.0     0.31    0.11     72.0       -0.02;
      -0.22   0.0     0.41    0.16    108.0       -0.02;
       0.0    0.35    0.25    0.21     90.0        0.01;
       0.0    0.1     0.046   0.046     0.0        0.01;
       0.0   -0.1     0.046   0.046     0.0        0.01;
      -0.08  -0.605   0.046   0.023     0.0        0.01;
       0.0   -0.605   0.023   0.023     0.0        0.01;
       0.06  -0.605   0.046   0.023    90.0        0.01];
##Shepp_Logan without skull
AI9 = [0.0      0.0     0.92    0.69     90.0     0.0;
        0.0     -0.0184  0.874   0.6624   90.0    0.2;
        0.22     0.0     0.31    0.11     72.0    -0.2;
       -0.22     0.0     0.41    0.16    108.0    -0.2;
        0.0      0.35    0.25    0.21     90.0     0.1;
        0.0      0.1     0.046   0.046     0.0     0.1;
        0.0     -0.1     0.046   0.046     0.0     0.1;
       -0.08    -0.605   0.046   0.023     0.0     0.1;
##     -0.08    -0.605   0.046   0.023     0.0    -0.5;
        0.0     -0.605   0.023   0.023     0.0     0.1;
        0.06    -0.605   0.046   0.023    90.0     0.5];



AI2 = [0.0    0.0     0.92    0.69     90.0        2.0;
       0.0   -0.0184  0.874   0.6624   90.0       -0.9;
       0.22   0.0     0.31    0.11     72.0       -0.1;
      -0.22   0.0     0.41    0.16    108.0       -0.1;
       0.0    0.35    0.25    0.21     90.0        0.3;
       0.0    0.1     0.046   0.046     0.0        0.3;
       0.0   -0.1     0.046   0.046     0.0        0.3;
      -0.08  -0.605   0.046   0.023     0.0        0.3;
       0.0   -0.605   0.023   0.023     0.0        0.3;
       0.06  -0.605   0.046   0.023     90.0       0.3];
      ## Shepp_Logan
AI2b = [0.0      0.0     0.92    0.69     90.0     1.0;
        0.0     -0.0184  0.874   0.6624   90.0    -0.8;
        0.22     0.0     0.31    0.11     72.0    -0.2;
       -0.22     0.0     0.41    0.16    108.0    -0.2;
        0.0      0.35    0.25    0.21     90.0     0.1;
        0.0      0.1     0.046   0.046     0.0     0.1;
        0.0     -0.1     0.046   0.046     0.0     0.1;
       -0.08    -0.605   0.046   0.023     0.0     0.1;
##     -0.08    -0.605   0.046   0.023     0.0    -0.5;
        0.0     -0.605   0.023   0.023     0.0     0.1;
        0.06    -0.605   0.046   0.023    90.0     0.5];
##      0.06  -0.605   0.046   0.023    90        0.1];
##  test 1: solid cylinder
AI3 = [0.0      0.0       0.7    0.7      0.0         2.0];

##  test 2: hollow cylinder
AI4 = [0.0      0.0       0.7    0.7      0.0         2.0;
       0.0      0.0       0.6    0.6      0.0        -2.0];

##  test 2: hollow cylinder with rod # Deathstar
AI5 = [0.0      0.0       0.7    0.7      0.0         2.0;
       0.0      0.0       0.6    0.6      0.0        -2.0;
       0.2      0.2       0.2    0.2      0.0         2.0];


xlow  = -1.0; #// image left
xhigh =  1.0; #// image right
ylow  = -1.0; #// image bottom
yhigh =  1.0; #// image top

xstep = 0.001; #// x resolution
ystep = 0.001; #// y resolution

##AI = AI1;
##AI = AI2; #//
##AI = AI1;
##AI = AI5;
##AI = AI2b;

xrange = xlow:xstep:xhigh;
yrange = ylow:ystep:yhigh;
nx = length(xrange);
ny = length(yrange);

#on crée les 2 phantoms (Deathstar et Shep_Logan)

x = phantom(AI5,xlow,xhigh,ylow,yhigh,xstep,ystep);  ##Vectorized bitmap
P1 = reshape(x,nx,ny);
y = phantom(AI2b,xlow,xhigh,ylow,yhigh,xstep,ystep);  ##Vectorized bitmap
P2 = reshape(y,nx,ny);
names = ["CTDeathstar", "CTShepp_Logan"]
#num = [128,256]
num = [128]
#num = [256]
variables = ["x","y"]
#counts = ["20k","100k","500k"]
#count256 = [2.75,13.75,68.5]
#count128 = [10.0,50.0,250.0]
counts = ["100","1000"]

imshow(P2,cmap=(ColorMap("gray")));


#string("$(names[1])_$(variables[1])$(num[1])_$(counts[1])")
X = []
Y = []

for j in num #pour les images de 128 et 256 pixel
    for k in 1:length(counts) # pour 22, 110 et 550k counts
        if j == 128
            X = imresize(P1,(j,j))
            X[X .< 0.0] = 0.0; #imresize creates stuff like -2.5 * 10^-15...
            println("$(names[1])_$(counts[k]) = les comptes sont ajoutés dans Create_noisy_sinogram2 ")
            Y = imresize(P2,(j,j))
            Y[Y.< 0.0]=0.0;
            println("$(names[2])_$(counts[k]) = les comptes sont ajoutés dans Create_noisy_sinogram2 ")
        else
            X = imresize(P1,(j,j))
            X[X .< 0.0] = 0.0
            println("$(names[1])_$(counts[k]) = les comptes sont ajoutés dans Create_noisy_sinogram2 ")
            Y = imresize(P2,(j,j))
            Y[Y.< 0.0]=0.0;
            println("$(names[2])_$(counts[k]) = les comptes sont ajoutés dans Create_noisy_sinogram2 ")
        end
        save(string("$(names[1])_$(variables[1])$(j)_$(counts[k]).jld"), string("$(variables[1])$(j)_$(counts[k])"), X)
        save(string("$(names[2])_$(variables[2])$(j)_$(counts[k]).jld"), string("$(variables[2])$(j)_$(counts[k])"), Y)
    end
end
