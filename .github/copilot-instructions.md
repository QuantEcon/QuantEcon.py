# QuantEcon.py Development Instructions

QuantEcon.py is a high-performance, open-source Python library for quantitative economics. The library provides tools for economics research including Markov chains, dynamic programming, game theory, quadrature, and optimization algorithms.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup (REQUIRED)
- **ALWAYS use conda environment**: `conda env create -f environment.yml` 
  - Takes 3-5 minutes to complete. NEVER CANCEL. Set timeout to 10+ minutes.
  - Creates environment named 'qe' with Python 3.13 and all dependencies
- **Activate environment**: `eval "$(conda shell.bash hook)" && conda activate qe`
- **Development install**: `flit install` 
  - Installs package in development mode for testing changes
  - Takes < 30 seconds

### Build and Test Workflow
- **Linting**: `flake8 --select F401,F405,E231 quantecon`
  - Takes < 30 seconds
  - Note: Repository has some existing style violations - this is expected
- **Full test suite**: `coverage run -m pytest quantecon`
  - Takes 5 minutes 11 seconds. NEVER CANCEL. Set timeout to 15+ minutes.
  - Runs 536 tests across all modules
  - All tests should pass with 2 warnings (expected)
- **Quick smoke test**: `python -c "import quantecon as qe; print('Version:', qe.__version__)"`
- **Package build**: `flit build`
  - Creates wheel and source distributions in dist/
  - Takes < 1 minute

### Validation Scenarios
After making changes, ALWAYS run these validation steps:
1. **Import test**: `python -c "import quantecon as qe; print('Import successful, version:', qe.__version__)"`
2. **Basic functionality test**:
```python
python -c "
from quantecon.markov import DiscreteDP
import numpy as np
R = np.array([[10, 8], [6, 4]])
Q = np.array([[[0.9, 0.1], [0.8, 0.2]], [[0.7, 0.3], [0.6, 0.4]]])
ddp = DiscreteDP(R, Q, 0.95)
result = ddp.solve(method='policy_iteration')
print('DiscreteDP test successful, policy:', result.sigma)
"
```
3. **Run relevant tests**: `pytest quantecon/tests/test_[module].py -v` for specific modules
4. **Run flake8 linting** before committing

## Key Project Structure

### Core Modules
- `quantecon/` - Main package source code
  - `markov/` - Markov chain and dynamic programming tools
  - `game_theory/` - Game theory algorithms and utilities  
  - `optimize/` - Optimization algorithms
  - `random/` - Random number generation utilities
  - `tests/` - Main test suite (536 tests total)

### Configuration Files
- `pyproject.toml` - Main project configuration using flit build system
- `environment.yml` - Conda environment specification with all dependencies
- `.github/workflows/ci.yml` - CI pipeline (tests on Python 3.11, 3.12, 3.13)
- `pytest.ini` - Test configuration including slow test markers

### Dependencies
Core runtime dependencies (auto-installed in conda env):
- `numba>=0.49.0` - JIT compilation for performance
- `numpy>=1.17.0` - Array operations
- `scipy>=1.5.0` - Scientific computing
- `sympy` - Symbolic mathematics
- `requests` - HTTP library

## Timing Expectations and Timeouts

**CRITICAL TIMING INFORMATION:**
- **Conda environment creation**: 3-5 minutes (timeout: 10+ minutes)
- **Full test suite**: 5 minutes 11 seconds (timeout: 15+ minutes) 
- **Package build**: < 30 seconds (timeout: 2 minutes)
- **Development install**: < 30 seconds (timeout: 2 minutes)
- **Linting**: < 30 seconds (timeout: 2 minutes)

**NEVER CANCEL these operations** - they may appear to hang but are processing large dependency trees or running comprehensive tests.

## Common Tasks

### Making Code Changes
1. Ensure conda environment is active: `conda activate qe`
2. Make your changes to files in `quantecon/`
3. Run development install: `flit install`
4. Test imports: `python -c "import quantecon as qe; print('Import OK')"`
5. Run relevant tests: `pytest quantecon/tests/test_[relevant_module].py`
6. Run linting: `flake8 --select F401,F405,E231 quantecon`

### Adding New Features
1. Add code to appropriate module in `quantecon/`
2. Add tests to `quantecon/tests/test_[module].py`
3. Update `quantecon/__init__.py` if exposing new public API
4. Run full test suite to ensure no regressions
5. Validate with example usage

### Debugging Failed Tests
1. Run specific test: `pytest quantecon/tests/test_[module].py::test_function -v`
2. Use pytest markers: `pytest -m "not slow"` to skip long-running tests
3. Check test output and traceback for specific failure modes
4. Many tests use numerical algorithms - check for convergence issues

## Important Notes

### CI/CD Pipeline
- GitHub Actions runs tests on Windows, Ubuntu, and macOS
- Tests Python 3.11, 3.12, and 3.13
- Includes flake8 linting and coverage reporting
- Publishing to PyPI is automated on git tags

### Network Limitations
- **pip install from PyPI may fail** due to network timeouts in sandboxed environments
- **Always use conda environment** for reliable dependency management
- Documentation building may fail due to missing dependencies - focus on code functionality

### Performance Considerations
- Many algorithms use numba JIT compilation - first run may be slower
- Test suite includes performance-sensitive numerical algorithms
- Some tests marked as "slow" - use `pytest -m "not slow"` to skip them during development

### Repository Status
- Current version: Check `quantecon/__init__.py` for `__version__` variable
- Build system: flit (modern Python packaging)
- License: MIT
- Documentation: ReadTheDocs (quanteconpy.readthedocs.io)

### Maintenance Notes
- When creating new releases, verify that timing expectations and test counts in these instructions remain accurate
- Version information is dynamically referenced to avoid hardcoded values

## Quick Reference Commands

```bash
# Setup (do once)
conda env create -f environment.yml
eval "$(conda shell.bash hook)" && conda activate qe

# Development workflow
flit install                                    # Install in development mode
python -c "import quantecon as qe; print(qe.__version__)"  # Test import
pytest quantecon/tests/test_[module].py        # Test specific module
flake8 --select F401,F405,E231 quantecon       # Lint code

# Full validation (before committing)
coverage run -m pytest quantecon               # Full test suite (5+ minutes)
flit build                                      # Build packages
```

Remember: This is a numerical computing library with complex dependencies. Always use the conda environment and expect longer build/test times than typical Python projects.