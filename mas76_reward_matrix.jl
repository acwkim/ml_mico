using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()

feature = CSV.read("../binkar/feature.csv",  DataFrame) ##Feature Matrix
xopt = CSV.read("../binkar/xopt.csv",  DataFrame) ##Optimal strategies for the instances we generated


###MIPLIB Files
coef = CSV.read("../mas76/coef.csv", DataFrame,header = false)
A = CSV.read("../mas76/A.csv",  DataFrame,header = false)
b = CSV.read("../mas76/b.csv",  DataFrame,header = false)
lb = CSV.read("../mas76/lb.csv",  DataFrame,header = false)
ub = CSV.read("../mas76/ub.csv", DataFrame,header = false)
intcon = CSV.read("../mas76/intcon.csv",  DataFrame,header = false)
###


intcon = convert(Array, intcon)
coef = convert(Array, coef)
A = convert(Array, A)
b = convert(Array, b)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)
b_center = convert(Array{Float64,2}, b)
c_center = deepcopy(coef)

c = coef




#Solve mas76 problem for a given parameter b and a strategy st
function mas_st(b,st,gurobi_env)
    n = size(A)[2]
    ni = length(intcon)
    nc = n - ni
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    z = st[1:ni]
    idx_1 = findall(!iszero, st[ni+1:ni+nc])
    idx_2 = findall(!iszero, st[ni+nc+1:ni+nc+nc])
    idx_3 = findall(!iszero, st[ni+nc+nc+1:end])
    A_idx = A[idx_3, :]
    b_idx = b[idx_3, :]

    @variable(final ,x[1:nc])

    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end

    for q ∈ idx_2
        @constraint(final, 1000000000000 >= x[q])
    end
    @constraint(final, A_idx[:, 1:ni] * z .+ A_idx[:, ni+1:n] * x .<= b_idx)
    @objective(final, Min, sum(c[i+ni] * x[i] for i = 1:nc) + sum(c[i] * z[i] for i = 1:ni));

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:ni] * z .+ A[:, ni+1:n] * opt_x .- 0.001 .>= b) || any(opt_x .- 0.01 .> 1000000000000)
            base_val = 1e6
        end
    end


    return base_val
end








x_opt = []
for i = 1:7000
    push!(x_opt, convert(Array,xopt[i,:]))
end

numu = length(unique(x_opt))

dic = countmap(x_opt)
sor = sort(collect(dic), by = x -> x[2], rev = true)

list = []
for i = 1:numu
    push!(list, sor[i][1])
end


Reward = ones(Float64, (1000, length(list)))

##


@time for l = 1:700

        d = convert(Array,feature[l,:])


        for k = 1:length(list)
            st = list[k]
            real_val = mas_st(d,st, gurobi_env)
            Reward[l,k] = real_val
        end
end

##

CSV.write("Reward.csv", convert(DataFrame,Reward))
