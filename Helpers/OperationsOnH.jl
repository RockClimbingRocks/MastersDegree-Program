module OperationsOnH
    using LinearAlgebra;


    function AverageValueOfOperator(H)
        return tr(H) / size(H)[1]
    end

    global function OperatorNorm(H) 
        return AverageValueOfOperator(H^2) - AverageValueOfOperator(H)^2   
    end

end