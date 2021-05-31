
# How to contribute to fxpmath

## **Bugs**

* **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/francof2a/fxpmath/issues).

* If you're unable to find an open issue addressing the problem, [open a new issue](https://github.com/francof2a/fxpmath/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.

* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

* Before submitting, it is highly recommended:
  * If the bug is not related with an issue, one issue must be opened.
  * If the bug has a linked issue, add a test function in [**tests/test_issues.py**](https://github.com/francof2a/fxpmath/blob/master/tests/test_issues.py) file. The name of the function must be `test_issue_{issue number}_{fxpmath version}`, for example: `test_issue_9_v0_3_6`.
  * Update **version** modifying __version__ at [__init__.py](https://github.com/francof2a/fxpmath/blob/master/fxpmath/__init__.py). The new version must be a release candidate, incrementing only PATCH field if there is not a release candidate and add a release candidate field; for example: '0.4.1' to '0.4.2-rc.0'. If actual version is a release candidate, you must increment only the _rc_ field, for example: '0.4.1-rc.3' to '0.4.1-rc.4'.
  * Run all tests and check all of them are succeful.
  * Update [changelog.txt](https://github.com/francof2a/fxpmath/blob/master/changelog.txt) at the top, keeping format. Link the solved issue.

Tests are executed using **pytest**.

## **New features or changes**

* Suggest your change in the [discussions](https://github.com/francof2a/fxpmath/discussions) and start writing code.

* Do not open an issue on GitHub until you have collected positive feedback about the change. GitHub issues are primarily intended for bug reports and fixes.

* Fork **fxpmath** repository.
* Check if a branch for the development of next version exists. If it exists, you should create a new brach from it, keeping the name of the brach as prefix; for example, if the development branch is `dev-0.5.0`, your new branch should be named `dev-0.5.0-{something}`. If it doesn't exist, create a branch with the name `dev-{new version}` incrementing MINOR version field; for example, if actual version is `0.4.1`, your new branch should be named `dev-0.5.0` or `dev-0.5.0-{something}`.
* Build an specific test file to test the new functionality or just update some of the test files with new test functions. All the tests are running executed using *pytest*.
* Run all tests and check all of them are succeful.
* Update **version** modifying __version__ at [__init__.py](https://github.com/francof2a/fxpmath/blob/master/fxpmath/__init__.py).
* Update [changelog.txt](https://github.com/francof2a/fxpmath/blob/master/changelog.txt) at the top, keeping format.
* Open a new GitHub pull request.

## **Do you have questions about the source code?**

* Ask any question about how to use **fxpmath** at [discussions](https://github.com/francof2a/fxpmath/discussions).

## **Do you want to contribute to the fxpmath documentation?**

* Fxpmath documentation site is [https://francof2a.github.io/fxpmath/](https://francof2a.github.io/fxpmath/).

* This site is build from [Github Pages branch](https://github.com/francof2a/fxpmath/tree/gh-pages). Documentation contributions must be made here.

* Is highly recommended to start a discussion in [discussions](https://github.com/francof2a/fxpmath/discussions) before begin writing about a topic.

It is very welcome review and update docstrings of classes, properties, methods and functions of fxpmath.

Thanks!
