# CONTRIBUTION_GUIDELINES.md

# Contribution Guidelines

This document defines the engineering standards, development workflow, coding principles, documentation rules, and collaboration guidelines for this project.

Every contributor, whether human or AI-assisted, should follow these guidelines to ensure the codebase remains clean, maintainable and production-ready.

---

# 1. Project Philosophy

This project is intended to be developed like a production software system rather than a college assignment.

The primary goals are:

* Clean Architecture
* Modular Design
* Maintainability
* Scalability
* Readability
* Consistency
* High Code Quality

Every implementation should prioritize long-term maintainability over short-term convenience.

---

# 2. Understand Before Changing

Before modifying any code:

* Explore the project structure.
* Read the relevant modules.
* Understand existing architecture.
* Understand current implementations.
* Identify existing utilities that can be reused.

Never begin coding immediately.

Understanding the codebase comes before implementation.

---

# 3. Respect Existing Architecture

Do not redesign the project unless explicitly requested.

Work with the existing architecture.

Prefer extending current implementations over replacing them.

If a significantly better design is found:

* Explain it.
* Discuss trade-offs.
* Wait for approval before refactoring.

---

# 4. Keep Changes Minimal

Only modify files directly related to the requested task.

Avoid:

* unnecessary formatting
* renaming unrelated files
* moving files
* changing APIs
* unrelated refactoring

Each change should have a clear purpose.

---

# 5. Respect Module Ownership

Assume different developers own different modules.

Do not modify another module unless absolutely required.

When integration is required:

* use existing interfaces
* avoid breaking existing functionality
* preserve backward compatibility whenever possible

---

# 6. Modular Design

Avoid placing excessive logic inside a single file.

Split responsibilities into dedicated modules.

Example:

* Model loading
* Inference
* Utilities
* Configuration
* Visualization
* Schemas

should all remain separate whenever practical.

---

# 7. Code Quality Standards

Follow:

* Python 3.11+
* PEP8
* Type Hints
* Docstrings
* Logging
* Error Handling

Prefer:

* reusable functions
* reusable classes
* dependency injection
* configuration files

Avoid:

* global variables
* duplicate logic
* hardcoded paths
* magic numbers

---

# 8. Explain Before Implementing

Before writing code:

Explain:

* what will change
* why it is needed
* which files will change
* architectural impact

Implementation should only begin after the approach is understood.

---

# 9. One Task at a Time

Only complete the requested task.

Do not anticipate future phases.

Do not implement future features.

After completing the requested work:

Stop.

Wait for review and approval.

---

# 10. Reuse Existing Code

Before creating:

* utility functions
* helper classes
* configuration
* schemas
* logging
* APIs

Always check whether an implementation already exists.

Reuse whenever possible.

---

# 11. Documentation Rules

Every meaningful change must update the project documentation.

Review and update when applicable:

* README.md
* CHANGELOG.md
* PROJECT_STATUS.md
* RESEARCH_LOG.md
* TODO.md

Documentation should always represent the current project state.

---

# 12. Testing

Every completed task should include verification.

When applicable:

* Unit Tests
* Integration Tests
* Smoke Tests

Code should not be considered complete without testing.

---

# 13. Preserve Consistency

Maintain:

* existing folder structure
* naming conventions
* documentation style
* logging style
* coding style
* architecture

New code should feel like it belongs to the existing project.

---

# 14. Handle Uncertainty Properly

If requirements are ambiguous:

Do not guess.

Ask for clarification before implementing.

---

# 15. Long-Term Maintainability

Write code as if another engineer will maintain it six months later.

Optimize for:

* readability
* simplicity
* maintainability

Avoid unnecessary complexity.

---

# 16. Architecture Evolution

The project architecture may evolve.

Libraries, models and approaches may change.

Design new implementations so that components are:

* modular
* replaceable
* loosely coupled

Future changes should require minimal modifications.

---

# 17. Working with AI Coding Assistants

AI coding assistants are tools used to accelerate development.

When using any AI coding assistant:

* Understand the existing codebase before generating code.
* Never overwrite unrelated code.
* Never perform large-scale refactoring unless requested.
* Limit modifications to the requested feature.
* Follow all project architecture rules.
* Update documentation after implementation.
* Keep generated code consistent with the existing project.

---

# 18. Documentation Tone

All documentation should be written from the perspective of the development team.

Do not mention:

* AI assistants
* ChatGPT
* Codex
* Cursor
* Claude
* Gemini
* Copilot

Avoid wording such as:

* "I created..."
* "I generated..."
* "As an AI..."
* "The assistant..."

Instead write documentation in a professional engineering style.

Examples:

Good:

* Implemented FastSAM integration.
* Added segmentation utilities.
* Updated configuration.

Avoid:

* I implemented...
* Generated by AI...
* Created using ChatGPT...

---

# 19. Git Workflow

Development should follow a feature-branch workflow.

Never commit directly to the main branch.

Recommended branch naming:

feature/<feature-name>

Examples:

feature/fastsam

feature/clip

feature/species-classifier

feature/depth-distance

Every feature should be developed independently.

Merge only after testing and review.

---

# 20. Commit Messages

Use meaningful commit messages.

Examples:

feat(segmentation): integrate FastSAM pipeline

feat(classification): implement CLIP classification

fix(mask): resolve contour extraction bug

docs: update phase documentation

refactor(config): simplify configuration loading

Avoid vague commit messages such as:

* Update
* Fix
* Changes
* Final Version

---

# 21. Project Documentation

The following project files should remain synchronized throughout development.

README.md

Project overview and setup.

CHANGELOG.md

History of implemented changes.

PROJECT_STATUS.md

Current development status.

RESEARCH_LOG.md

Research findings, experiments and architectural decisions.

TODO.md

Pending tasks and future work.

---

# 22. Phase Completion Checklist

A phase is considered complete only after:

□ Implementation completed

□ Tests passing

□ Documentation updated

□ CHANGELOG updated

□ PROJECT_STATUS updated

□ RESEARCH_LOG updated

□ TODO updated

□ README updated (if applicable)

□ Code reviewed

□ Commit created

□ Repository pushed

---

# 23. Final Principle

Always contribute as a professional software engineer.

Prioritize:

Understanding

↓

Planning

↓

Implementation

↓

Testing

↓

Documentation



The objective is not only to build a working system, but to build one that is maintainable, scalable, and understandable for every future contributor.
