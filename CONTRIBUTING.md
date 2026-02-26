# Contributing to MLX Swift

We want to make contributing to this project as easy and transparent as
possible.

## Build from Source
We use git submodules to organize [mlx](https://github.com/ml-explore/mlx), [mlx-c](https://github.com/ml-explore/mlx-c), and [mlx-swift](https://github.com/ml-explore/mlx-swift) as a single project.

1. Clone mlx-swift `git clone git@github.com:ml-explore/mlx-swift.git`
1. Init submodules `git submodule init`
1. Update submodules `git submodule update --recursive`
1. Open your clonded `mlx-swift` folder in Xcode to build, edit, and debug

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Every PR should have passing tests and at least one review.
5. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.

   You can also run the formatters manually as follows:

     ```shell
     swift-format -i Source/MLX/*.swift
     ```

   or run `pre-commit run --all-files` to check all files in the repo.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
