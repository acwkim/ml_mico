using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()

A = CSV.read("../bea/A.csv", DataFrame, header = false)
b = CSV.read("../bea/b.csv", DataFrame, header = false)
coef = CSV.read("../bea/coef.csv",DataFrame,  header = false)
Aeq = CSV.read("../bea/Aeq.csv", DataFrame, header = false)
beq = CSV.read("../bea/beq.csv", DataFrame, header = false)
lb = CSV.read("../bea/lb.csv", DataFrame, header = false)
ub = CSV.read("../bea/ub.csv", DataFrame, header = false)
intcon = CSV.read("../bea/intcon.csv", DataFrame, header = false)

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
coef = convert(Array, coef)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)




#Solve beasley problem for a given parameter c
function beas(c,gurobi_env)
    n = size(A)[2]
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:1250] >=0)
    @variable(final, z[1:1250], Bin)
    @constraint(final, A[:, 1:1250] * x .+ A[:, 1251:n] * z .<= b)
    @constraint(final, Aeq[:, 1:1250] * x .+ Aeq[:, 1251:n] * z .== beq)
    @objective(final, Min, sum(c[i+1250] * z[i] for i = 1:1250));

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.0001
    c2 = abs.(A[:, 1:1250] * xx .+ A[:, 1251:n] * zz .- b) .<= 0.0001

    tight = vcat(zz,c1,c2)

    return base_val, tight
end


#Solve beasley problem for a given parameter c and a strategy st
function beas_st(c,st, gurobi_env)

    m,n = size(A)[1], size(A)[2]


    integer = st[1:1250]
    idx_1 = findall(!iszero, st[1251:1250+n])
    idx_2 = findall(!iszero, st[1251+n:end])



    A_idx = A[idx_2, :]
    b_idx = b[idx_2, :]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n-1250])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    @variable(final, z[1:1250])
    @constraint(final, z .== integer)
    @constraint(final, A_idx[:, 1:1250] * x .+ A_idx[:, 1251:n] * z .<= b_idx)
    @constraint(final, Aeq[:, 1:1250] * x .+ Aeq[:, 1251:n] * z .== beq)
    @objective(final, Min, sum(c[i] * z[i] for i = 1:1250))

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


#Generate s_num data, from a ball with center c_center and radius r
function sample_generating(r,c_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()


    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((50 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        dn = dn[1:50] .* r
        c = deepcopy(c_center)
        for i = 1:50
            c[i+1250] = c[i+1250] + dn[i]
        end
        feat = c[1251:1300]

        opt_val, tight = beas(c,gurobi_env)


        columns = reshape(feat, (1, 50))
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