using JLD2;
using LinearAlgebra;
using SparseArrays;
# using PyPlot;

include("../../HAMILTONIANS/ManyBody/WithoutInteractions/SYK2.jl");
using .SYK2;
include("../../HAMILTONIANS/ManyBody/WithoutInteractions/SYK2LOC.jl");
using .SYK2LOC;
include("../../HAMILTONIANS/ManyBody/WithInteractions/SYK4.jl");
using .SYK4;
include("../../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_CM.jl");
using .SYK4LOC_CM;
include("../../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_MINMAX.jl");
using .SYK4LOC_MINMAX;
include("../../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_ROK.jl");
using .SYK4LOC_ROK;

include("../../HAMILTONIANS/ManyBody/c+c-.jl");
using .c⁺c;
include("../../HAMILTONIANS/ManyBody/c+c+c-c-.jl");
using .c⁺c⁺cc;

include("../../PATHS/Paths.jl");
using .PATH;

include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;
include("../GetInteractionStrengths.jl");
using .IntStrength;


function GetEta(L::Int64)
    GetEtaDict = Dict(4 => 0.1, 6 => 0.1, 8 => 0.15, 10 => 0.2);
    return L <= 10 ? GetEtaDict[L] : 0.25;
end




function GetChaosIndicators(L:: Int64, a::Float64, b::Float64, H2:: Module, H4:: Module, n:: Int64, λ:: Float64, τs::Vector{Float64})
    η = GetEta(L);
    
    Es′s= Vector{Vector{Float64}}(undef,n);
    r′s = Vector{Float64}(undef,n);
    Ks  = Vector{Vector{Float64}}(undef,n);
    Kcs = Vector{Vector{Float64}}(undef,n);


    D = binomial(L, Int(L÷2));
    H = Matrix{Float64}(undef, D, D);
    # ϕ = Matrix{Float64}(undef, D, D);
    for i in 1:n
        param2 = H2.Params(L,a,b);
        param4 = H4.Params(L,a,b);

        H .= Matrix( c⁺c.Ĥ(L, param2.t) .+ λ .* c⁺c⁺cc.Ĥ(L, param4.U))
        # Es′s[i], ϕ = eigen(Symmetric(Matrix( H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));

        Es′s[i] = eigvals(Symmetric(H));
        r′s[i] = MBCI.r̂(Es′s[i], η);
    end


    Ks, Kcs = MBCI.K̂(Es′s, τs);


    dir = PATH.GetDirectoryIntoDataFolder(@__FILE__);
    subDir = PATH.GetSubDirectory("MB");
    fileName = PATH.GetFileName(H2, H4, L, N);
    groupPath = PATH.GetGroupPath(λ, a, b);

    jldopen("$(dir)$(subDir)$(fileName)", "a+") do file
        # folder["$(groupPath)/Info"] = "η = $(η),  τs = $(τs)";
        file["$(groupPath)/Es′s"] = Es′s;
        file["$(groupPath)/r′s"]  = r′s;
        file["$(groupPath)/Ks"]   = Ks;
        file["$(groupPath)/Kcs"]  = Kcs;
        file["$(groupPath)/τs"]   = τs;
    end
end



L = parse(Int64, ARGS[1]);
N = parse(Int64, ARGS[2]);
# L=6;
# N=1;

H2 = SYK2LOC;
H4 = SYK4LOC_ROK;

as = [0.5, 0.75, 1., 1.25];
bs = [0.05];

τs =  10 .^ LinRange(-4, 1, 400)
λ′s = IntStrength.λ̂′s();


println("Running program name is: ", split(PROGRAM_FILE, "MastersDegree-Program/")[end])

println(PROGRAM_FILE)
println(@__FILE__);


println("L = $(L),    N = $(N)");
# println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")


dir = PATH.GetDirectoryIntoDataFolder(@__FILE__);
subDir = PATH.GetSubDirectory("MB");
fileName = PATH.GetFileName(H2, H4, L, N);

for b in bs 
    for a in as
        for (i,λ) in enumerate(λ′s)
            
            isDataAllreadyCalculated::Bool = false;
            jldopen("$(dir)$(subDir)$(fileName)", "a+") do file
                groupPath = PATH.GetGroupPath(λ, a, b);
                isDataAllreadyCalculated = haskey(file, "$(groupPath)");
            end
            
            isDataAllreadyCalculated ? continue : nothing;
            @time GetChaosIndicators(L, a, b, H2, H4, N, λ, τs);
        end
    end
end
