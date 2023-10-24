using Distributed;
using JLD2;
@everywhere using LinearAlgebra;
@everywhere using SparseArrays;


@everywhere include("../../HAMILTONIANS/ManyBody/SYK2.jl");
using .SYK2;
@everywhere include("../../HAMILTONIANS/ManyBody/SYK4.jl");
using .SYK4;
@everywhere include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
@everywhere include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;


function GetDirectory()
    dir = "../../DATA/";
    return dir;
end


function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end



@everywhere struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end


@everywhere function GetEta(L::Int64)
    GetEtaDict = Dict(6 => 0.1, 8 => 0.15, 10 => 0.2);

    return L <= 10 ? GetEtaDict[L] : 0.25;
end


@everywhere function GetChaosIndicators0(L:: Int64, n:: Int64, q:: Float64):: Vector{ReturnObject}
    # println("Get n=$n CIs on processor p=$(myid()) and thread t=$(Threads.threadid())");
    
    setA:: Vector{Int64} = collect(1:1:L÷2);
    setB:: Vector{Int64} = collect(L÷2+1:1:L);
    indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);


    η = GetEta(L);
    
    returnObject = Vector{ReturnObject}(undef, n)
    for i in 1:n
        F = eigen(Symmetric(Matrix(H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));

        returnObject[i] = ReturnObject(
            F.values,
            (CI.LevelSpacingRatio(F.values, η), CI.InformationalEntropy(F.vectors, η), CI.EntanglementEntropy(F.vectors, setA, setB, indices, η))
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
        ci   = GetChaosIndicators0(L, n′s[i], q);
        
        i₁ = sum(n′s[1:i-1]) + 1; 
        i₂ = i₁ + n′s[i] -1 ; 
        lock(lck) do 
            returnObject[i₁:i₂] .= ci; 
        end 
    end 

    return returnObject;
end

function (L:: Int64, a::Float64, b::Float64, H2:: Module, H4:: Module, N:: Int64, λ:: Float64, τs::Vector{Float64})
    η = GetEta(L);
    
    Es′s= Vector{Vector{Float64}}(undef,N);
    r′s = Vector{Float64}(undef,N);
    Ks  = Vector{Vector{Float64}}(undef,N);
    Kcs = Vector{Vector{Float64}}(undef,N);


    D = binomial(L, Int(L÷2));
    H = Matrix{Float64}(undef, D, D);
    # ϕ = Matrix{Float64}(undef, D, D);

    p:: Int64 = nworkers();
    n:: Int64 = Int(N÷p);
    o:: Int64 = Int(N - n*p);
    n′s:: Vector{Int64} = [n for _ in 1:p]; 
    n′s[1] += o;

    println(n′s);


    #DO STUFFFF
    # xx = @async Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p)
    # map(x -> append!(returnObject, x), @sync Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p));
    # println("Normalizatioins are calculated");



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

τs =  10 .^ LinRange(-4, 1, 400);

λ′s = IntStrength.λ̂′s();
reverse!(λ′s);


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
            @time GetChaosIndicators_MPandMT(L, a, b, H2, H4, N, λ, τs);
        end
    end
end



for (i,q) in enumerate(q′s)
    @time GetChaosIndicators_MPandMT(L, N, q);
end

