using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()


##These files are MIPLIB files
A = CSV.read("../binkar/A.csv",DataFrame, header = false)
b = CSV.read("../binkar/b.csv", DataFrame,header = false)
coef = CSV.read("../binkar/coef.csv",DataFrame, header = false)
Aeq = CSV.read("../binkar/Aeq.csv", DataFrame,header = false)
beq = CSV.read("../binkar/beq.csv", DataFrame,header = false)
lb = CSV.read("../binkar/lb.csv", DataFrame,header = false)
ub = CSV.read("../binkar/ub.csv",DataFrame, header = false)
intcon = CSV.read("../binkar/intcon.csv",DataFrame, header = false)

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
coef = convert(Array, coef)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)


b_center = deepcopy(b)


#Solve binkar problem for a given parameter b
function binkar(b,gurobi_env)
    n = size(A)[2]


    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n-170] >=0)
    @variable(final, z[1:170], Bin)
    @constraint(final, A[:, 1:2128] * x .+ A[:, 2129:n] * z .<= b)
    @constraint(final, Aeq[:, 1:2128] * x .+ Aeq[:, 2129:n] * z .== beq)
    @objective(final, Min, sum(coef[i] * x[i] for i = 1:2128) + sum(coef[i+2128] * z[i] for i = 1:170))

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.0001
    c2 = abs.(A[:, 1:2128] * xx .+ A[:, 2129:n] * zz .- b) .<= 0.0001

    tight = vcat(zz,c1,c2)

    return base_val, tight
end


#Generate s_num samples, from a ball with center c_center and radius r
function sample_generating(r,b_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()

    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((10 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        b = (dn[1:10] .* r) .+ b_center  #This is the sampled f

        opt_val, tight = binkar(b,gurobi_env)


        columns = reshape(b, (1, 10))
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

@time feature, opt_x, optval = sample_generating(1,b_center, 900, gurobi_env)

CSV.write("feature.csv", feature)
CSV.write("x.csv", opt_x)
CSV.write("val.csv", optval)
