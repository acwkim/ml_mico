using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()


##MIPLIB Files
A = CSV.read("../neos3/A.csv", DataFrame, header = false)
b = CSV.read("../neos3/b.csv", DataFrame,header = false)
coef = CSV.read("../neos3/coef.csv", DataFrame,header = false)
Aeq = CSV.read("../neos3/Aeq.csv",  DataFrame,header = false)
beq = CSV.read("../neos3/beq.csv",  DataFrame,header = false)
lb = CSV.read("../neos3/lb.csv",  DataFrame,header = false)
ub = CSV.read("../neos3/ub.csv", DataFrame,header = false)
intcon = CSV.read("../neos3/intcon.csv",  DataFrame,header = false)
##MIPLIb Files

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
coef = convert(Array, coef)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)
b_center = convert(Array{Float64,2}, b)

c = coef


#Solve neos problem for a given parameter b 
function neos(b,gurobi_env)
    n = size(A)[2]
    ni = length(intcon)
    nc = n - ni
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final , x[1:nc] >=0)
    @variable(final, z[1:ni], Bin)
    @constraint(final, A[:, 1:nc] * x .+ A[:, nc+1:n] * z .<= b)
    @constraint(final, Aeq[:, 1:nc] * x .+ Aeq[:, nc+1:n] * z .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:nc) + sum(c[i+nc] * z[i] for i = 1:ni));

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.01
    c2 = abs.(A[:, 1:nc] * xx .+ A[:, nc+1:n] * zz .- b) .<= 0.01

    tight = vcat(zz,c1,c2)

    return base_val, tight
end

#Solve binkar problem for a given parameter feat and a strategy st
function neos_st(feat,st, gurobi_env)

    m,n = size(A)[1], size(A)[2]
    n = size(A)[2]
    ni = length(intcon)
    nc = n - ni

    b[1:15] = feat

    integer = st[1:ni]
    idx_1 = findall(!iszero, st[ni+1:ni+nc])
    idx_2 = findall(!iszero, st[ni+nc+1:end])


    A_idx = A[idx_2, :]
    b_idx = b[idx_2, :]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:nc])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    @constraint(final, A_idx[:, 1:nc] * x .+ A_idx[:, nc+1:n] * integer .<= b_idx)
    @constraint(final, Aeq[:, 1:nc] * x .+ Aeq[:, nc+1:n] * integer .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:nc) + sum(c[i+nc] * integer[i] for i = 1:ni));

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:nc] * opt_x .+ A[:, nc+1:n] * integer .- 0.001 .>= b)
            base_val = 1e6
        end
    end





    return base_val
end


#Generate s_num samples, from a ball with center b_center and radius r
function sample_generating(r,b_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()
    n = size(A)[2]

    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((15 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        dn = dn[1:15] .* r
        b = deepcopy(b_center)
        b[1:15] = b[1:15] .+ dn
        feat = b[1:15]


        opt_val, tight = neos(b,gurobi_env)


        columns = reshape(feat, (1, 15))
        tight = round.(Int, tight)
        tight = reshape(tight, (1,length(tight)))
        val = reshape([opt_val], (1,1))

        df1 = convert(DataFrame, columns)
        df2 = convert(DataFrame, tight)
        df3 = convert(DataFrame, val)

        append!(feature, df1)
        append!(opt_x, df2)
        append!(optval, df3)
        if i % 100 == 0
            println(i)
        end
    end
    return feature, opt_x, optval
end


