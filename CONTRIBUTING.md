# Contributing to [Project Name]

Thank you for your interest in contributing to stera-sdk! We welcome community contributions, whether they are bug reports, feature requests, or code improvements. 

To maintain high code quality and smooth collaboration, please follow these guidelines.

---

## 🐛 Bug Reports

Before submitting a bug report, please check the existing [Issues](https://github.com/fpv-labs/stera-sdk/issues) to ensure it hasn't already been reported.

When opening a new bug report, please use our **Bug Report Template** and provide the following details:

* **Summary:** A clear and concise description of what the bug is.
* **Environment:** * OS / Hardware: [e.g., Ubuntu 22.04, Apple M2]
    * Version/Commit SHA: [e.g., v1.2.0 or `a1b2c3d`]
* **Steps to Reproduce:** Clear steps to reproduce the behavior. 
    1. Go to '...'
    2. Click on '...'
    3. Run command '...'
* **Expected Behavior:** A clear description of what you expected to happen.
* **Actual Behavior / Logs:** Include any error logs, stack traces, or screenshots. Wrap code/logs in triple backticks (\`\`\`).
* **Minimal Reproducible Example (MRE):** If applicable, provide a short snippet of code or a minimal dataset that reproduces the issue.

---

## 💡 Feature Requests

We love hearing ideas for new features or infrastructure improvements! To propose a feature, please open an Issue and include:

* **Problem Statement:** Is your feature request related to a problem? (e.g., "I'm frustrated when...")
* **Proposed Solution:** A clear and concise description of what you want to happen.
* **Alternatives Considered:** A description of any alternative solutions or workarounds you've considered.
* **Additional Context:** Any architectural diagrams, mockups, or benchmarks that help explain your proposal.

---

## 🚀 Pull Request (PR) Guidelines

We welcome contributions from the community to make this a vibrant open source effort. Its recommended to communicate your plan for feature development, in the same format as a feature request template to explain what you plan to do, or initiate some upfront conversation through our discord (fpvlabs.ai/discord). To get your PR reviewed and merged quickly, please adhere to the following workflow:

### 1. Branch Strategy & Scoping
* **One Feature Per PR:** Keep PRs atomic. Do not bundle unrelated changes, bug fixes, and formatting updates into a single PR.
* **Branch Naming:** Use descriptive branch names:
    * `fix/issue-description`
    * `feat/feature-name`
    * `docs/update-readme`

### 2. Code Quality & Formatting
* **Linting & Style:** Ensure your code complies with our style guides. Run local linters before committing (e.g., `black`, `flake8`, `clang-format`).
* **Type Hinting:** (If applicable) Ensure all new functions have proper type hints.

### 3. Testing
* **Proof of correctness:** Ensure that the code is correct and provide results that validate that the new code works. This can be in the form of tests that can be written, qualitative results that can be showcased etc..
* Include commands on  how the new code can be tested

### 4. Documentation
* Update the `README.md` or inline docstrings (e.g., Google or Sphinx style) if you are changing user-facing APIs, adding configurations, or introducing new modules.

---

## 📝 Pull Request Checklist

When you open a PR, please use the following checklist in the description field:

```markdown
### PR Description
- **Related Issue:** Fixes #[Issue Number]
- **Summary of Changes:** [Briefly describe what this PR introduces]

### Checklist
- [ ] My code follows the code style and formatting guidelines of this project.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have made corresponding changes to the documentation.
- [ ] I have added tests that prove my fix is effective or that my feature works.
