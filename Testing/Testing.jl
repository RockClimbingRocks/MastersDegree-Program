module MastersDegree_Helpers

    struct FermionAlgebra
        L::Int
        S::Int
        
        function FermionAlgebra(L::Int, S::Int)
            new(L, S)
        end

        global function GetNumberOfConfigurationsOfOneSpin(fa:: FermionAlgebra)
            println("22")
            return Int((2 * fa.S + 1) ^ fa.L)
        end    
    end    

end



fa = MastersDegree_Helpers.FermionAlgebra(3, 2)
println(fa)
n_configs = MastersDegree_Helpers.GetNumberOfConfigurationsOfOneSpin(fa)
println(n_configs)  # Output: 729