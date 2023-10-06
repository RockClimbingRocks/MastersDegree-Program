using LinearAlgebra;
using Distributed;
using SparseArrays;
using BenchmarkTools;


@everywhere include("../../Hamiltonians/H2.jl");
using .H2;
@everywhere include("../../Hamiltonians/H4.jl");
using .H4;
# @everywhere include("../Helpers/ChaosIndicators/ChaosIndicators.jl");
# using .ChaosIndicators;


@everywhere include("../../Helpers/FermionAlgebra.jl");
using .FermionAlgebra;


@everywhere include("../../Helpers/ChaosIndicators/ChaosIndicatorsFromSpectrume.jl");
using .CI;


function GetDirectory()
    dir = "./";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end




@everywhere function GetChaosIndicators0(L:: Int64, n:: Int64, q:: Float64):: Vector{ReturnObject}
    # println("Get n=$n CIs on processor p=$(myid()) and thread t=$(Threads.threadid())");
    
    setA:: Vector{Int64} = collect(1:1:L÷2);
    setB:: Vector{Int64} = collect(L÷2+1:1:L);
    indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);

    
    returnObject = Vector{ReturnObject}()
    for i in 1:n
        H₂ = H2.Ĥ(H2.Params(L));
        H₄ = H4.Ĥ(H4.Params(L));
        H = @. H₂ + q * H₄;
        F = eigen(Symmetric(Matrix(H)));

        push!(returnObject, 
            ReturnObject(
                F.values,
                (CI.LevelSpacingRatio(F.values), CI.InformationalEntropy(F.vectors), CI.EntanglementEntropy(F.vectors, setA, setB, indices))
            )
        );
    end

    return returnObject;
end



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

    println(n′s);
    

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

    println(n′s);

    returnObject = Vector{ReturnObject}();

    println("---------------")
    # xx = @async Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p)
    map(x -> append!(returnObject, x), @sync Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p));
    
    println("Chaos indicators are calculated");
    display(returnObject);


    # dir = GetDirectory();
    # fileName = GetFileName(L, N, q);
    
    # folder = jldopen("$(dir)$(fileName).jld2", "w");
    # folder["Es_and_CIs"]=returnObject;
    # close(folder);
end


L = 6;
N = 5;
q = 0.5;

println("   workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

GetChaosIndicators_MPandMT(L, N, q);

