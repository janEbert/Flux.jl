using Documenter, Flux, NNlib

DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
makedocs(modules=[Flux, NNlib],
         doctest = true,
         sitename = "Flux",
         format = Documenter.HTML(
                 analytics = "UA-36890222-9",
                 assets = ["assets/flux.css"],
                 prettyurls = get(ENV, "CI", nothing) == "true",
         ),
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
                  "Internals" =>
                    ["Backpropagation" => "internals/tracker.md"],
                  "Community" => "community.md"])

deploydocs(repo = "github.com/FluxML/Flux.jl.git")

