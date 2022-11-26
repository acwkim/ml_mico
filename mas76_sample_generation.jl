using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()


##MIPLIB Files
coef = CSV.read("../mas76/coef.csv", DataFrame,header = false)
A = CSV.read("../mas76/A.csv",  DataFrame,header = false)
b = CSV.read("../mas76/b.csv",  DataFrame,header = false)
lb = CSV.read("../mas76/lb.csv",  DataFrame,header = false)
ub = CSV.read("../mas76/ub.csv", DataFrame,header = false)
intcon = CSV.read("../mas76/intcon.csv",  DataFrame,header = false)
##


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


#Solve mas76 problem for a given parameter b 
function mas(b,gurobi_env)
    n = size(A)[2]
    ni = length(intcon)
    nc = n - ni
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    set_optimizer_attribute(final, "MIPGap", 0.005)
@variable(final ,1000000000000 >= x[1:nc] >= 0)
@variable(final, z[1:ni], Bin)
@constraint(final, A[:, 1:ni] * z .+ A[:, ni+1:n] * x .<= b)
# @constraint(final, Aeq[:, 1:nc] * x .+ Aeq[:, nc+1:n] * z .== beq)
@objective(final, Min, sum(c[i+ni] * x[i] for i = 1:nc) + sum(c[i] * z[i] for i = 1:ni));

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.01
    c2 = abs.(xx .- 1000000000000) .<= 0.01
    c3 = abs.(A[:, 1:ni] * zz .+ A[:, ni+1:n] * xx .- b) .<= 0.01

    tight = vcat(zz,c1,c2,c3)

    return base_val, Int.(tight)
end



#Generate s_num samples, from a ball with center b_center and radius r
function sample_generating(r,b_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()


    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((12 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        dn = dn[1:12] .* r
        b = deepcopy(b_center)
        b = b .+ dn
        feat = b


        opt_val, tight = mas(b,gurobi_env)


        columns = reshape(feat, (1, 12))
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


feature, opt_x, optval = sample_generating(2, b_center, 900, gurobi_env)

CSV.write("feature.csv", feature)
CSV.write("x.csv", opt_x)
CSV.write("val.csv", optval)
