module SYK4
    using Distributions;

    struct Params
        L:: Int64;
        U:: Array{Float64, 4}; 
        S:: Float64;

        function Params(L:: Int64, U̲::Union{Array{Float64}, Missing} = missing, S::Float64=1/2)            
            if isequal(U̲, missing)
                # println("---")
                U = zeros(Float64, (L,L,L,L))

                for i=1:L, j=i+1:L, k=1:L, l=k+1:L
                    if U[i,j,k,l] == 0 && i!=j && k!=L
                        U[i,j,k,l] = rand(Normal(0, 1));
                        U[j,i,l,k] = U[k,l,i,j] = U[l,k,j,i] = U[i,j,k,l];
                        U[i,j,l,k] = U[j,i,k,l] = U[k,l,j,i] = U[l,k,i,j] = -U[i,j,k,l];
                    end
                end
            else
                U = U̲[1:L, 1:L, 1:L, 1:L]
            end

            new(L, U, S)
        end
    end


    function AnaliticalNormOfHamiltonianAveraged(L::Int64)
        norm2 = L*(L-2)*(L^2 + 6L + 8 - 2*(L-2)/(L-1)) / 4 ;
        return √norm2;
    end

end