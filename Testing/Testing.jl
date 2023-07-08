using SparseArrays;
using LinearAlgebra;
# using Combinatorics
# using ITensors
using Statistics
using BenchmarkTools;
using JLD2;

include("../Helpers/FermionAlgebra.jl");
using .FermionAlgebra;

include("../Hamiltonians/H2.jl");
using .H2;

include("../Hamiltonians/H4.jl");
using .H4;



# a = randn(5000,5000)


# display(@benchmark begin a[1000:4000,1000:4000] = randn(3001,3001) end;);
# println()
# println()

# display(@benchmark begin b = @view a; b[1000:4000,1000:4000] = randn(3001,3001) end)




# folder = jldopen("aaaa.jld2", "w");
#     folder["Es"] = Es;
#     folder["rs"] = rs;
#     folder["Ss"] = Ss;
#     folder["EEs"] = EEs;
#     folder["Ks"] = Ks;
#     folder["τs"] = τs;
#     folder["K̄s"] = Ks_smooth;
#     folder["τ̄s"] = τs_smooth;
#     folder["Kcs"] = Kcs;
#     folder["τ_th"] = τ_Th;
#     folder["t_H"] = t_H;
#     folder["t_Th"] = t_Th;
#     folder["g"] = g;
# close(folder);      

L′s = [1,2,3,4];
η′s = [0.1,0.2];


# jldopen("example.jld2", "w") do file
#     for L in L′s
#         MyGroupL = JLD2.Group(file, "L=$L");
#         MyGroupL["xxx"] = 42;
#     end
# end

# jldopen("exampleqssss.jld2", "w") do file
#     for L in L′s
#         file["L=$L/xxx1/xx"] = 44;
#         file["L=$L/xxx2/xx"] = 45;
#     end
# end

# jldopen("example.jld2", "r") do file
#     for L in L′s
#         # MyGroupL = JLD2.Group(file, "L=$L");
#         xxx1 = file["L=$L/xxx1/xx"];
#         xxx2 = file["L=$L/xxx2/xx"];

#         println(xxx1);
#         println(xxx2);
#     end
# end


# jldopen("example.jld2", "w") do file
#     for L in L′s
#         MyGroupL = JLD2.Group(file, "L=$L");

#         MyGroupL["xxx"] = 42;
#     end
# end

