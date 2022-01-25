import Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("GLM")
Pkg.add("StatsBase")
Pkg.add("Lathe")
Pkg.add("MLbase")
Pkg.add("ClassImbalance")
Pkg.add("ROCAnalysis")
using Pkg
using DataFrames
using CSV
using Plots
using GLM
using StatsBase
using Lathe
using MLBase
using ClassImbalance
using ROCAnalysis
ENV["columns"] = 1000
df  = CSV.read("dataset.csv",DataFrame)
println(size(df))
describe(df)
names(df)
Lathe.preprocess.OneHotEncode(df,:Geography)
Lathe.preprocess.OneHotEncode(df,:Gender)
select!(df, Not([:RowNumber, :CustomerId,:Surname,:Geography,:Gender,:Male]))
first(df,5)
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df,.75);
println(train)
# Train logistic regression model
fm = @formula(Exited ~ CreditScore + Age + Tenure + 
              Balance + NumOfProducts + HasCrCard + 
              IsActiveMember + EstimatedSalary)
logit = glm(fm, train, Binomial(), ProbitLink())

prediction = predict(logit, test)