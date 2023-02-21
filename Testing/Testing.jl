module MastersDegree_Helpers

    struct FermionAlgebra
        L::Int
        S::Int
        
        function FermionAlgebra(L::Int, S::Int)
            new(L, S)
        end
        
        global function GetN1umberOfConfigurationsOfOneSpin()
            return Int((2 * S + 1) ^ L)
        end
    end
end



fa = MastersDegree_Helpers.FermionAlgebra(3, 2)
n_configs = fa.GetNumberOfConfigurationsOfOneSpin()
println(n_configs)  # Output: 729