
# To še raymisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module ChaosIndicators
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics

    include("./FermionAlgebra.jl");
    using .FermionAlgebra;

    function Ĥₕ(L, S, hᵢ)
        Hₕ = spzeros(Float64, Int(2*S+1)^L, Int(2*S+1)^L)
        nᵢ = fill(sparse(1.0I,Int(2*S+1),Int(2*S+1)), L); 
        n  = sparse(1.0I,Int(2*S+1),Int(2*S+1));  n[1,1] = 0
        nᵢ[1] = n
   
        for k in 1:L
             Hₕ += hᵢ[k].*foldl(kron,circshift(nᵢ,k-1))
        end
        return Hₕ
   end

    global function LevelSpacingRatio(H₀, params, W̲, numberOfIterations)
        r_of_W = zeros(Float64, length(W̲))
        ind = FermionAlgebra.IndecesOfSubBlock(params.L, params.S);
        # println(length(Indeces))
   
        # First we compute Hamiltonian for W=0, so then we only need to calculate the part with magnetic field. We only consider part with Sᶻ=0
        # Σλ₀ = tr(H₀) 
        # Id = sparse(I, length(ind), length(ind))
   
        # Here we go throu every value of W
        for i in range(1, length(W̲))
             print("-",i) 
   
             #---------------- In this for loop we average over more calculations of "r" for diffrent uniform distributions of disorer "W". 
             for j in 1:numberOfIterations
                  Hₕ  = Ĥₕ(params.L, params.S, (2*W̲[i]).*rand(params.L) .- W̲[i])[ind,ind]
                  
                #   Σλₕ = tr(Hₕ)
                #   λ̄   =  (Σλ₀ + Σλₕ)/length(ind)
                  # λ = @time  sort(eigs(H₀.+Hₕ .- λ̄.*Id, nev = Nₑᵢᵧ, ritzvec= false, which=:SM )[1] ) # Tuki dej rajs nev=500
                  λ = eigvals( Matrix(H₀ .+ Hₕ) )[(2*length(ind))÷5 : (3*length(ind))÷5]
   
                  δ = λ[2:end] .- λ[1:end-1]
                  r = map((x,y) -> min(x,y)/max(x,y), δ[1:end-1], δ[2:end] )
                  r_of_W[i] += sum(r) / length(r)
             end 
        end
        return r_of_W ./ numberOfIterations
   end




   global function InformationalEntropy(H₀, params, W̲, numberOfIterations)
        ind = FermionAlgebra.IndecesOfSubBlock(params.L, params.S);

        ∑Sₘ = zeros(Float64, length(W̲))
        for i in range(1,length(W̲))
            print("-",i)
            for j in 1:numberOfIterations 
                Hₕ = Ĥₕ(params.L, params.S, (2*W̲[i]).*rand(params.L) .- W̲[i])[ind,ind]
                ϕ = eigvecs( Matrix(H₀ .+ Hₕ) )
                ϕ  = ϕ[(2*length(ind))÷5 : (3*size(H₀,1))÷5,(2*size(H₀,1))÷5 : (3*size(H₀,1))÷5]
                Sₘ = sum( map(x -> - abs(x)^2 * log(abs(x)^2), ϕ) , dims=1)

                ∑Sₘ[i] += sum(Sₘ)/length(Sₘ)
            end 
        end
        return ∑Sₘ./numberOfIterations
    end

end




