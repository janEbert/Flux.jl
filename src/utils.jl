# Arrays
"""
    glorot_uniform(dims...)

Return an `Array` of size `dims` containing random variables taken from a uniform
distribution in the interval ``[-x, x]``, where `x = sqrt(24 / sum(dims)) / 2`.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_uniform(2, 3)
2×3 Array{Float32,2}:
 0.601094  -0.57414   -0.814925
 0.900868   0.805994   0.057514
```
"""
glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))

"""
    glorot_normal(dims...)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `(2 / sum(dims))`.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_normal(3, 2)
3×2 Array{Float32,2}:
  0.429505  -0.0852891
  0.523935   0.371009
 -0.223261   0.188052
```
"""
glorot_normal(dims...) = randn(Float32, dims...) .* sqrt(2.0f0/sum(dims))

ones(T::Type, dims...) = Base.ones(T, dims...)
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones(dims...) = Base.ones(Float32, dims...)
zeros(dims...) = Base.zeros(Float32, dims...)

"""
    unsqueeze(xs, dim)

Return `xs` reshaped into an `Array` one dimensionality higher than `xs`,
where `dim` indicates in which dimension `xs` is extended.

# Examples
```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.unsqueeze(xs, 1)
1×3 Array{Array{Int64,1},2}:
 [1, 2]  [3, 4]  [5, 6]

julia> Flux.unsqueeze([1 2; 3 4], 2)
2×1×2 Array{Int64,3}:
[:, :, 1] =
 1
 3

[:, :, 2] =
 2
 4
```
"""
unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

"""
    stack(xs, dim)

Concatenate the given `Array` of `Array`s `xs` into a single `Array` along the
given dimension `dim`.

# Examples
```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.stack(xs, 1)
3×2 Array{Int64,2}:
 1  2
 3  4
 5  6

julia> cat(xs, dims=1)
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
```
"""
stack(xs, dim) = cat(unsqueeze.(xs, dim)..., dims=dim)

"""
    unstack(xs, dim)

Unroll the given `xs` into an `Array` of `Array`s along the given dimension `dim`.

# Examples
```jldoctest
julia> Flux.unstack([1 3 5 7; 2 4 6 8], 2)
4-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
```
"""
unstack(xs, dim) = [copy(selectdim(xs, dim, i)) for i in 1:size(xs, dim)]

"""
    chunk(xs, n)

Split `xs` into `n` parts.

# Examples
```jldoctest
julia> Flux.chunk(1:10, 3)
3-element Array{Array{Int64,1},1}:
 [1, 2, 3, 4]
 [5, 6, 7, 8]
 [9, 10]
```
"""
chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

batchindex(xs, i) = (reverse(Base.tail(reverse(axes(xs))))..., i)

"""
    frequencies(xs)

Count the number of times that each element of `xs` appears.

# Examples
```jldoctest
julia> Flux.frequencies(['a','b','b'])
Dict{Char,Int64} with 2 entries:
  'a' => 1
  'b' => 2
```
"""
function frequencies(xs)
  fs = Dict{eltype(xs),Int}()
  for x in xs
    fs[x] = get(fs, x, 0) + 1
  end
  return fs
end

head(x::Tuple) = reverse(Base.tail(reverse(x)))

squeezebatch(x) = reshape(x, head(size(x)))

"""
  batch(xs)

Batch the arrays in `xs` into a single array.

# Examples
```jldoctest
julia> Flux.batch([[1,2,3],[4,5,6]])
3×2 Array{Int64,2}:
 1  4
 2  5
 3  6
```
"""
function batch(xs)
  data = first(xs) isa AbstractArray ?
    similar(first(xs), size(first(xs))..., length(xs)) :
    Vector{eltype(xs)}(undef, length(xs))
  for (i, x) in enumerate(xs)
    data[batchindex(data, i)...] = x
  end
  return data
end

"""
Return the given sequence padded with `p` up to a maximum length of `n`.

# Examples
```jldoctest
julia> rpad([1, 2], 4, 0)
4-element Array{Int64,1}:
 1
 2
 0
 0

julia> rpad([1, 2, 3], 2, 0)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
Base.rpad(v::AbstractVector, n::Integer, p) = [v; fill(p, max(n - length(v), 0))]

"""
    batchseq(seqs, pad)

Take a list of `N` sequences, and turn them into a single sequence where each
item is a batch of `N`. Short sequences will be padded by `pad`.

# Examples
```jldoctest
julia> Flux.batchseq([[1, 2, 3], [4, 5]], 0)
3-element Array{Array{Int64,1},1}:
 [1, 4]
 [2, 5]
 [3, 0]
```
"""
function batchseq(xs, pad = nothing, n = maximum(length(x) for x in xs))
  xs_ = [rpad(x, n, pad) for x in xs]
  [batch([xs_[j][i] for j = 1:length(xs_)]) for i = 1:n]
end

# Other

"""
    throttle(f, timeout; leading=true, trailing=false)

Return a function that when invoked, will only be triggered at most once
during `timeout` seconds.

Normally, the throttled function will run as much as it can, without ever
going more than once per `wait` duration; but if you'd like to disable the
execution on the leading edge, pass `leading=false`. To enable execution on
the trailing edge, pass `trailing=true`.
"""
function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing
  result = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        result = f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @async try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> (result = f(args...; kwargs...))
    end

    return result
  end
end

"""
    @jit ...

The `@jit` annotation can be applied to any code, and the code will be compiled
for performance.

    @jit f(x) = @jit(x) + @jit(x)

Note that compilation happens regardless of the `@jit` macro, so it should only
be used for aesthetic purposes, or by recovering Python users.
"""
macro jit(ex)
  esc(ex)
end
