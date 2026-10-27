# Contributing to QuantEcon.py

Thanks for your interest in contributing to QuantEcon.py.

The contribution guide is maintained in the project documentation:

- [Rendered guide on Read the Docs](https://quanteconpy.readthedocs.io/en/latest/contributing.html)
- [Source file in this repository](docs/source/contributing.rst)

The guide covers where to start, how to set up a development environment,
and expectations for tests and documentation.

## Code of AI Use

QuantEcon welcomes AI-assisted contributions under its Code of AI Use,
[QEP-5](https://github.com/QuantEcon/qeps/blob/main/qeps/qep-0005-code-of-ai-use.md). The Code asks that a human chooses the task, reads the result and answers review on it,
and that every pull request says whether AI tools were used.

**Disclose in one line.** Put one `Assisted-by:` line in the pull-request description or in a
commit message:

- `Assisted-by: Claude Code (Claude Fable 5.1)` when AI tools meaningfully contributed to the
  code, tests or text, naming the tool and the model;
- `Assisted-by: none` when they did not.

A `Co-Authored-By:` trailer that a tool adds to a commit on its own, as Claude Code and Copilot
do, also counts. The line is a marker, not a narrative: it names what was used, not what it did.
Opening a pull request confirms that a human chose the task, read the result, and will answer
review on it.

**If the line is missing.** On a pull request from outside the organisation, an automated
comment asks for the line and a reply, once, and applies the `author-response` label. Nothing is
blocked; a maintainer reads the reply.

The Code also asks that issues labelled `good first issue` are worked without AI tools.
