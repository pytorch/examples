# Contributing to examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

We actively welcome your pull requests.

If you're new, we encourage you to take a look at issues tagged with [good first issue](https://github.com/pytorch/examples/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

### For new examples

0. Create a GitHub issue proposing a new example and make sure it's substantially different from an existing one.
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests to `run_python_examples.sh`.
3. Create a `README.md`.
4. Add a card with a brief description of your example and link to the repo to
   the `docs/source/index.rst` file and build the docs by running:

   ```
   cd docs
   virtualenv venv
   source venv/bin/activate
   pip install -r requirements.txt
   make html
   ```

   When done working with `virtualenv`, run `deactivate`.

5. Verify that there are no issues in your doc build. You can check the preview locally
   by installing [sphinx-serve](https://pypi.org/project/sphinx-serve/)
   then running `sphinx-serve -b build`.
6. Ensure your test passes locally.
7. If you haven't already, complete the Contributor License Agreement ("CLA").
8. Address any feedback in code review promptly.

## For bug fixes

1. Fork the repo and create your branch from `main`.
2. Make sure you have a GPU-enabled machine, either locally or in the cloud. `g4dn.4xlarge` is a good starting point on AWS.
3. Make your code change.
4. First, install all dependencies with `./run_python_examples.sh "install_deps"`.
5. Then, make sure that `./run_python_examples.sh` passes locally by running the script end to end.
6. If you haven't already, complete the Contributor License Agreement ("CLA").
7. Address any feedback in code review promptly.

## Contributor License Agreement ("CLA")

To accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
