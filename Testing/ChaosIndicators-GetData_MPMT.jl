using Distributed;
@everywhere using LinearAlgebra;
@everywhere using SparseArrays;
using JLD2;

@everywhere include("../Hamiltonians/H2.jl");
using .H2;
@everywhere include("../Hamiltonians/H4.jl");
using .H4;
@everywhere include("../Helpers/FermionAlgebra.jl");
using .FermionAlgebra;
@everywhere include("../Helpers/ChaosIndicators/ChaosIndicatorsFromSpectrume.jl");
using .CI;



function GetDirectory()
    dir = "./";
    return dir;
end


function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end




# @everywhere CIs_type = Tuple{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}};

@everywhere struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end





@everywhere function GetChaosIndicators(L:: Int64, n:: Int64, q:: Float64):: Vector{ReturnObject}
    # println("Get n=$n CIs on processor p=$(myid()) and thread t=$(Threads.threadid())");

    function Create_H′s(H′s_ch:: Channel{SparseMatrixCSC{Float64}})
        for i in 1:n
            # println("   Creating H ", i)
            H₂ = H2.Ĥ(H2.Params(L));
            H₄ = H4.Ĥ(H4.Params(L));
            H = @. H₂ + q * H₄
            put!(H′s_ch, H);
        end
    end
    
    function DiagonalizeH′s(F′s_ch:: Channel{Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}})
        for H in H′s_ch
            # println("   Diagonalizing");
            F = eigen(Symmetric(Matrix(H)));
            put!(F′s_ch, F);
        end
    end
    
    H′s_ch = Channel{SparseMatrixCSC{Float64}}(Create_H′s);
    F′s_ch = Channel{Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}}(DiagonalizeH′s); 
    
    setA:: Vector{Int64} = collect(1:1:L÷2);
    setB:: Vector{Int64} = collect(L÷2+1:1:L);
    indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);
    
    
    returnObject = Vector{ReturnObject}()
    for F in F′s_ch
        push!(returnObject, 
            ReturnObject(
                F.values,
                (CI.LevelSpacingRatio(F.values), CI.InformationalEntropy(F.vectors), CI.EntanglementEntropy(F.vectors, setA, setB, indices))
            )
        );
    end

    return returnObject;
end

@everywhere function GetChaosIndicators_mt(L:: Int64, N:: Int64, q:: Float64)
    t:: Int64 = Threads.nthreads();
    n:: Int64 = Int(N÷t);
    o:: Int64 = Int(N - n*t);
    n′s:: Vector{Int64} = [n for _ in 1:t]; 
    n′s[1] += o;

    returnObject = Vector{ReturnObject}(undef, N);
    lck = Threads.ReentrantLock();

    # println(n′s)
    Threads.@threads for i = 1:t
        # println("   $i    threadid = ", Threads.threadid());
        ci   = GetChaosIndicators(L, n′s[i], q);
        
        i₁ = sum(n′s[1:i-1]) + 1; 
        i₂ = i₁ + n′s[i] -1 ; 
        lock(lck) do 
            returnObject[i₁:i₂] .= ci; 
        end 
    end 

    return returnObject;
end





function GetChaosIndicators_MPandMT(L:: Int, N:: Int, q:: Float64)
    p:: Int64 = nworkers();
    n:: Int64 = Int(N÷p);
    o:: Int64 = Int(N - n*p);
    n′s:: Vector{Int64} = [n for _ in 1:p]; 
    n′s[1] += o;

    returnObject = Vector{ReturnObject}();

    println("---------------")
    # xx = @async Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p)
    map(x -> append!(returnObject, x), @sync Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p));
    println("Normalizatioins are calculated");


    dir = GetDirectory();
    fileName = GetFileName(L, N, q);
    
    folder = jldopen("$(dir)$(fileName).jld2", "w");
    folder["Es_and_CIs"]=returnObject;
    close(folder);
end


# L = parse(Int64, ARGS[1]);
# N = parse(Int64, ARGS[2]);

# qmin:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
# qmax:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
# ρq:: Int64 = 6;


# Nq:: Int64 = Int((qmax - qmin)*ρq +1)
# x = LinRange(qmin, qmax, Nq);
# q′s = round.(10 .^(x), digits=5);

L = 6;
N = 10;
q = 0.5;



println("L = $(L),    N = $(N)");
println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

println("   workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

GetChaosIndicators_MPandMT(L, N, q);




