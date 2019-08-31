using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Flux, NNlib

DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
makedocs(modules=[Flux, NNlib],
         doctest = true,
         sitename = "Flux",
         pages = ["Home" => "index.md",
                  "Building Models" =>
                    ["Basics" => "models/basics.md",
                     "Recurrence" => "models/recurrence.md",
                     "Regularisation" => "models/regularisation.md",
                     "Model Reference" => "models/layers.md"],
                  "Training Models" =>
                    ["Optimisers" => "training/optimisers.md",
                     "Loss Functions" => "training/loss_functions.md",
                     "Training" => "training/training.md"],
                  "One-Hot Encoding" => "data/onehot.md",
                  "GPU Support" => "gpu.md",
                  "Saving & Loading" => "saving.md",
                  "Utility Functions" => "utilities.md",
                  "Performance Tips" => "performance.md",
                  "Datasets" => "datasets.md",
                  "Community" => "community.md"],
         format = Documenter.HTML(
             assets = ["assets/flux.css"],
             analytics = "UA-36890222-9",
             prettyurls = haskey(ENV, "CI")))

deploydocs(repo = "github.com/FluxML/Flux.jl.git")

