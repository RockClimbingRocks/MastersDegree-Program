using ITensors

# let
#   i = Index(3)
#   j = Index(5)
#   k = Index(2)
#   l = Index(7)

#   A = ITensor(i,j,k)
#   B = ITensor(j,l)

#   # Set elements of A
#   A[i=>1,j=>1,k=>1] = 11.1
#   A[i=>2,j=>1,k=>2] = -21.2
#   A[k=>1,i=>3,j=>1] = 31.1  # can provide Index values in any order
#   # ...

#   # Contract over shared index j
#   C = A * B

#   @show hasinds(C,i,k,l) # = true

#   D = randomITensor(k,j,i) # ITensor with random elements

#   # Add two ITensors
#   # must have same set of indices
#   # but can be in any order
#   R = A + D

#   nothing
# end





# i = Index(10)           # index of dimension 10
# j = Index(20)           # index of dimension 20
# M = randomITensor(i,j)  # random matrix, indices i,j
# U,S,V = svd(M)        # compute SVD with i as row index



# @show U      
# @show S      
# @show V      
ITensors.enable_debug_checks()

# let 
#     N = 10
#     sites = siteinds("S=1",N)

#     @show sites[1]

#     # Input operator terms which define
#     # a Hamiltonian matrix, and convert
#     # these terms to an MPO tensor network
#     # (here we make the 1D Heisenberg model)
#     os = OpSum()
#     for j=1:N-1
#         os += "Sz",j,"Sz",j+1
#         os += 0.5,"S+",j,"S-",j+1
#         os += 0.5,"S-",j,"S+",j+1
#     end
#     H = MPO(os,sites)

#     @show H

#     # Create an initial random matrix product state
#     psi0 = randomMPS(sites)

#     # Plan to do 5 passes or 'sweeps' of DMRG,
#     # setting maximum MPS internal dimensions
#     # for each sweep and maximum truncation cutoff
#     # used when adapting internal dimensions:
#     nsweeps = 5
#     maxdim = [10,20,100,100,200]
#     cutoff = 1E-10

#     # Run the DMRG algorithm, returning energy
#     # (dominant eigenvalue) and optimized MPS
#     energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
#     println("Final energy = $energy")
# end



let 
  N= 4
  s = "S=1/2";
  sites = siteinds(s,N)


  a⁺_op = [0 0; 1 0]

  ITensors.op(::OpName"a⁺",::SiteType"S=1/2") = [0 0; 1 0];
  ITensors.op(::OpName"a", ::SiteType"S=1/2") = [0 1; 0 0];

  os = opSum()
  for i in 1:N
    os += "a⁺", i,  "a", i+1 
    os += "a", i,  "a⁺", i+1 
  end

  H = MPO(os,sites);

  @show H;

  

end