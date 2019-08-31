using NNlib: logsoftmax, logσ

# Cost functions

"""
    mse(ŷ, y)

Mean square error between ŷ and y;
defined as ``\\frac{1}{n} \\sum_{i=1}^n (ŷ_i - y_i)^2``.

# Examples
```jldoctest
julia> Flux.mse([0, 2], [1, 1])
1//1
```
"""
mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)

"""
    crossentropy(ŷ, y; weight = 1)

Return the cross entropy between the given probability distributions.

# Examples
```jldoctest
julia> Flux.crossentropy(softmax([-1.1491, 0.8619, 0.3127]), [1, 1, 0])
3.085467254747739
```
"""
function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)
end

"""
    logitcrossentropy(logŷ, y; weight = 1)

`logitcrossentropy(logŷ, y)` is mathematically equivalent to
[`Flux.crossentropy(softmax(logŷ), y)`](@ref) but it is more numerically stable.

# Examples
```jldoctest
julia> Flux.logitcrossentropy([-1.1491, 0.8619, 0.3127], [1, 1, 0])
3.085467254747738
```
"""
function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) * 1 // size(y, 2)
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return `-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)`. The ϵ term provides numerical stability.

    julia> binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0.])
    3-element Array{Float64,1}:
    1.4244
    0.352317
    0.86167
"""
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(logŷ, y)

`logitbinarycrossentropy(logŷ, y)` is mathematically equivalent to `binarycrossentropy(σ(logŷ), y)`
but it is more numerically stable.

    julia> logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0.])
    3-element Array{Float64,1}:
     1.4244
     0.352317
     0.86167
"""
logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

"""
    normalise(x::AbstractArray; dims=1)

    Normalises x to mean 0 and standard deviation 1, across the dimensions given by dims. Defaults to normalising over columns.
"""
function normalise(x::AbstractArray; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end

function normalise(x::AbstractArray, dims)
  Base.depwarn("`normalise(x::AbstractArray, dims)` is deprecated, use `normalise(a, dims=dims)` instead.", :normalise)
  normalise(x, dims = dims)
end
