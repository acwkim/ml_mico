using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()

feature = CSV.read("../binkar/feature.csv",  DataFrame) #feature matrix to use for reward matrix computing.
xopt = CSV.read("../binkar/xopt.csv",  DataFrame)#optimal strategies to use for reward matrix computing.

####### These files are MIPLIB files
A = CSV.read("../binkar/A.csv",DataFrame, header = false)
b = CSV.read("../binkar/b.csv", DataFrame,header = false)
c = CSV.read("../binkar/coef.csv",DataFrame, header = false)
Aeq = CSV.read("../binkar/Aeq.csv", DataFrame,header = false)
beq = CSV.read("../binkar/beq.csv", DataFrame,header = false)
lb = CSV.read("../binkar/lb.csv", DataFrame,header = false)
ub = CSV.read("../binkar/ub.csv",DataFrame, header = false)
intcon = CSV.read("../binkar/intcon.csv",DataFrame, header = false)
#######

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
c = convert(Array, c)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)


b_center = deepcopy(b)




#Solve binkar problem for a given parameter b and a strategy st
function binkar_st(b,st, gurobi_env)
    # solve binkar problem applying a strategy st
    # penalty is set to 1e6 in this case.
    m,n = size(A)[1], size(A)[2]


    integer = st[1:170]
    idx_1 = findall(!iszero, st[171:n])
    idx_2 = findall(!iszero, st[172+n:end])



    A_idx = A[idx_2, :]
    b_idx = b[idx_2, :]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n-170])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    @variable(final, z[1:170])
    @constraint(final, z .== integer)
    @constraint(final, A_idx[:, 1:2128] * x .+ A_idx[:, 2129:n] * z .<= b_idx)
    @constraint(final, Aeq[:, 1:2128] * x .+ Aeq[:, 2129:n] * z .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:2128) + sum(c[i+2128] * z[i] for i = 1:170))

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:2128] * opt_x .+ A[:, 2129:n] * integer .- 0.001 .>= b)
            base_val = 1e6
        end
    end





    return base_val
end








x_opt = []
for i = 1:7000
    push!(x_opt, convert(Array,xopt[i,:]))
end

dic = countmap(x_opt)
sor = sort(collect(dic), by = x -> x[2], rev = true)

list = []
for i = 1:length(unique(x_opt))
    push!(list, sor[i][1])
end


Reward = ones(Float64, (7000, length(list)))

##


for l = 1:7000

        d = convert(Array,feature[l,:])


        for k = 1:length(list)
            st = list[k]
            real_val = binkar_st(d,st, gurobi_env)
            Reward[l,k] = real_val
        end
end

##

CSV.write("Reward.csv", convert(DataFrame,Reward))
