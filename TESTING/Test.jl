d=5
a = rand(d,d)

display(a)

@time map!(x -> x^2,a,a)
display(a)
# println(b)