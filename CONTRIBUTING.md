# Contribution Guidelines

Thank you for your interest in contributing to **CrossTL**!

## How to Contribute

### 1. Fork the Repository

Start by forking the **crosstl** repository on GitHub to your own account.

### 2. Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/your-username/crosstl.git
cd crosstl
```

### 3. Create a Branch

Create a new branch for your changes. Use a descriptive name for your branch:

```bash
git checkout -b feature-name
```

### 4. Make Changes

Make your changes to the codebase. Follow the project's coding style and guidelines. Ensure that your changes do not break existing functionality.

### 5. Write Unit Tests

Ensure that your code is reliable by writing unit tests:
- Add tests for any new functionality or changes.
- Make sure your code passes all existing and new tests.
- You can run the test suite locally using:

```bash
pytest tests/
```

### 6. Commit Your Changes

Commit your changes with a clear and concise commit message:

```bash
git add .
git commit -m "Description of your changes"
```

### 7. Push to GitHub

Push your changes to your forked repository on GitHub:

```bash
git push origin feature-name
```

### 8. Open a Pull Request

Open a pull request on the original repository. Provide a detailed description of your changes, why they are necessary, and any additional information that may be useful for the reviewers.

### Assigning or Creating Issues

If you're looking for something to work on, check out the open issues on our repository! You can assign yourself to an issue by commenting:

```plaintext
@CrossGL-issue-bot assign me
```

The bot will automatically assign the issue to you.

If you'd like to propose a new feature or report a bug, create a new issue with as much detail as possible, then comment with the same command to get it assigned.

## Contribution Pipeline

### Issue Tracking

We use GitHub Issues to track bugs, enhancements, and tasks. When opening an issue:
- Provide a clear description.
- Include steps to reproduce the issue if it's a bug.
- Suggest a solution if possible.

### Code Review

All contributions go through a code review process. The maintainers will review your pull request and provide feedback. Be prepared to make additional changes based on the feedback.

### Continuous Integration (CI)

We use CI to run automated tests on all pull requests. Ensure that your code passes all the tests before submitting.

### Merging

Once your pull request has been reviewed and approved, it will be merged into the main branch.

## Thank You

Thank you for contributing to **Crosstl**!
