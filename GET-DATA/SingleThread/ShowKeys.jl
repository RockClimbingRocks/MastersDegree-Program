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




# function ShowKeys(L:: Int64, a::Float64, b::Float64, H2:: Module, H4:: Module)
#     dir = PATH.GetDirectoryIntoDataFolder(@__FILE__);
#     subDir = PATH.GetSubDirectory("MB");
#     fileName = PATH.GetFileName(H2, H4, L, N);
#     groupPath = PATH.GetGroupPath(λ, a, b);

#     println()

#     jldopen("$(dir)$(subDir)$(fileName)", "a+") do file
#         # folder["$(groupPath)/Info"] = "η = $(η),  τs = $(τs)";
#         file["$(groupPath)/Es′s"] = Es′s;
#         file["$(groupPath)/r′s"]  = r′s;
#         file["$(groupPath)/Ks"]   = Ks;
#         file["$(groupPath)/Kcs"]  = Kcs;
#         file["$(groupPath)/τs"]   = τs;
#     end
# end



# L = parse(Int64, ARGS[1]);
# N = parse(Int64, ARGS[2]);
L=12;
N=3000;

H2 = SYK2LOC;
H4 = SYK4LOC_CM;

as = [0.5, 0.75, 1., 1.25];
bs = [0.05];

τs =  10 .^ LinRange(-4, 1, 400)
λ′s = IntStrength.λ̂′s();


println("Running program name is ", split(PROGRAM_FILE, "MastersDegree-Program/")[end])

println(PROGRAM_FILE)
println(@__FILE__);


println("L = $(L),    N = $(N)");
# println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

for b in bs 
    for a in as

        dir = PATH.GetDirectoryIntoDataFolder(@__FILE__);
        subDir = PATH.GetSubDirectory("MB");
        fileName = PATH.GetFileName(H2, H4, L, N);
        # groupPath = PATH.GetGroupPath(λ, a, b);
        # "/λ=$(λ)/a=$(a)/b=$(b)"
        # println()

        isDataAllreadyCalculated::Bool = false
    
        jldopen("$(dir)$(subDir)$(fileName)", "a+") do file       
            for (i,λ) in enumerate(λ′s)
                x = keys(file["/λ=$(λ)"]);
                println(x)
            end

            # x = keys(file);
            # println(x)
        end


    end
end
