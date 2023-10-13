module PATH
    """
    Function returns path into DATA folder. Directory path does not end with `/`.
    """
    function GetDirectoryIntoDataFolder(filePath:: AbstractString)
        path1 = split(filePath, "MastersDegree-Program")[end];
        println(path1)
        numberOfGoingBacks = count(x -> x=="/", split("Dpa/th/1", "")) - 1; 
        println(numberOfGoingBacks)
        
        relativePath = "../"^numberOfGoingBacks;

        relativePath *= "DATA";
        return relativePath;        
    end


    """
    Function returns path from DATA folder for OB or MB subfolder. Directory does not end with `/`.

    # Arguments
    - `MB_or_OB:: String`: Input parameter should be `MB` or `OB`, to determin subfolder of DATA folder.
    """
    function GetSubDirectory(MB_or_OB:: String)
        subDir = ""
        if MB_or_OB == "MB"
            subDir = "/ChaosIndicators-ManyBody";
        elseif MB_or_OB == "OB"
            subDir = "/ChaosIndicators-OneBody";
        else
            throw(error("Input should be either MB or OB."));
        end

        return subDir;        
    end


    """
    Function returns name of the saved file. Directory does not end with `/`.

    # Arguments 
    - `H2:: Module`: Module of non interacting Hamiltonian.
    - `H4:: Module`: Module of interacting Hamiltonian.
    - `L:: Int64`: Length of a system.
    - `N:: Int64`: Number of iterations.
    """
    function GetFileName(H2:: Module, H4:: Module, L:: Int64, N:: Int64)
        H2_module = split("$(H2)", ".")[2];
        H4_module = split("$(H4)", ".")[2];
        
        fileName = "/$(H2_module)->$(H4_module)_L$(L)_N$(N).jld2";
        return fileName;
    end


    
    """
    Function returns group path of the saved file. Path does not end with `/`.

    # Arguments 
    - `λ:: Float64`: Interaction strength.
    - `a:: Float64`: Parameter for detmening interaction decay.
    - `b:: Float64`: Parameter for detmening interaction decay.
    """
    function GetGroupPath(λ:: Float64, a:: Float64, b:: Float64)
        groupPath = "/λ=$(λ)/a=$(a)/b=$(b)";
        return groupPath
    end
end
