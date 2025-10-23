"""
Improvements for Polyglot Agent based on top-performing agent patterns.
These improvements focus on:
1. Git checkpoint management for state tracking
2. Enhanced test comparison (before/after state)
3. Multi-solution generation with consensus
4. Test runner auto-detection
5. Better syntax validation
"""

import subprocess
import os
import logging
import re
import ast
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# GIT CHECKPOINT MANAGEMENT
# ============================================================================

class GitCheckpointManager:
    """
    Manages git checkpoints for tracking code state and comparing test results.
    Key benefits:
    - Save state before making changes
    - Compare test results before/after changes
    - Revert to known-good states
    """

    @staticmethod
    def create_checkpoint(repo_path: str, checkpoint_name: str) -> dict:
        """
        Create a checkpoint (git commit + tag) at current state.
        Useful for saving initial state before making changes.

        Returns:
            dict with status, checkpoint_name, commit_hash, message
        """
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {
                    "status": "error",
                    "message": f"Not a git repository: {repo_path}"
                }

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if checkpoint already exists
            result = subprocess.run(
                ["git", "tag", "-l", checkpoint_name],
                capture_output=True,
                text=True,
                check=False
            )

            if result.stdout.strip():
                os.chdir(original_dir)
                return {
                    "status": "error",
                    "message": f"Checkpoint '{checkpoint_name}' already exists"
                }

            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )

            # Create commit if needed
            if status_result.stdout.strip():
                subprocess.run(
                    ["git", "commit", "-m", f"Checkpoint: {checkpoint_name}"],
                    capture_output=True,
                    text=True,
                    check=True
                )

            # Get current commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = hash_result.stdout.strip()

            # Create tag
            subprocess.run(
                ["git", "tag", checkpoint_name, commit_hash],
                check=True,
                capture_output=True
            )

            os.chdir(original_dir)

            return {
                "status": "success",
                "checkpoint_name": checkpoint_name,
                "commit_hash": commit_hash,
                "message": f"Checkpoint '{checkpoint_name}' created at {commit_hash[:8]}"
            }

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"
            }
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }

    @staticmethod
    def switch_checkpoint(repo_path: str, checkpoint_name: str, save_current: bool = True) -> dict:
        """
        Switch to a specific checkpoint, optionally stashing current changes.
        Useful for comparing test results between states.

        Returns:
            dict with status, checkpoint_name, commit_hash, message, stashed flag
        """
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {
                    "status": "error",
                    "message": f"Not a git repository: {repo_path}"
                }

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if checkpoint exists
            tag_result = subprocess.run(
                ["git", "tag", "-l", checkpoint_name],
                capture_output=True,
                text=True,
                check=False
            )

            if not tag_result.stdout.strip():
                os.chdir(original_dir)
                return {
                    "status": "error",
                    "message": f"Checkpoint '{checkpoint_name}' not found"
                }

            # Save current state if requested
            stashed = False
            if save_current:
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                if status_result.stdout.strip():
                    subprocess.run(
                        ["git", "stash", "push", "-u", "-m", f"Auto-stash before switching to {checkpoint_name}"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    stashed = True

            # Checkout the checkpoint
            subprocess.run(
                ["git", "checkout", checkpoint_name],
                capture_output=True,
                text=True,
                check=True
            )

            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            commit_hash = hash_result.stdout.strip()

            os.chdir(original_dir)

            return {
                "status": "success",
                "checkpoint_name": checkpoint_name,
                "commit_hash": commit_hash,
                "message": f"Switched to checkpoint '{checkpoint_name}' at {commit_hash[:8]}",
                "stashed": stashed
            }

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"
            }
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }

    @staticmethod
    def restore_stashed_changes(repo_path: str, stash_index: int = 0, remove_after_apply: bool = True) -> dict:
        """
        Restore previously stashed changes.

        Returns:
            dict with status, message, stash_index, removed flag
        """
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {
                    "status": "error",
                    "message": f"Not a git repository: {repo_path}"
                }

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if there are stashes
            stash_list_result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            if not stash_list_result.stdout.strip():
                os.chdir(original_dir)
                return {
                    "status": "error",
                    "message": "No stashed changes found"
                }

            stash_count = len(stash_list_result.stdout.strip().split('\n'))

            if stash_index >= stash_count:
                os.chdir(original_dir)
                return {
                    "status": "error",
                    "message": f"Stash index {stash_index} out of range. Only {stash_count} stash(es) available."
                }

            # Apply or pop the stash
            stash_ref = f"stash@{{{stash_index}}}"

            if remove_after_apply:
                command = ["git", "stash", "pop", stash_ref]
                action = "popped"
            else:
                command = ["git", "stash", "apply", stash_ref]
                action = "applied"

            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            os.chdir(original_dir)

            return {
                "status": "success",
                "message": f"Successfully {action} stash@{{{stash_index}}}",
                "stash_index": stash_index,
                "removed": remove_after_apply
            }

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Git stash command failed: {e.stderr if e.stderr else str(e)}"
            }
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }


# ============================================================================
# ENHANCED TEST COMPARISON
# ============================================================================

class EnhancedTestComparator:
    """
    Compares test results before and after changes to detect:
    - New failures (tests that passed before but fail now)
    - New passes (tests that failed before but pass now)
    - Regressions (passing tests that now fail - critical!)
    """

    @staticmethod
    def parse_test_output(output: str) -> Dict[str, List[str]]:
        """
        Parse test output to extract passed and failed tests.

        Returns:
            dict with 'passed' and 'failed' lists of test names
        """
        passed = []
        failed = []

        # Parse pytest output
        if "PASSED" in output or "FAILED" in output:
            for line in output.split('\n'):
                if " PASSED" in line:
                    # Extract test name
                    match = re.search(r'(test_\w+|::test_\w+)', line)
                    if match:
                        passed.append(match.group(1))
                elif " FAILED" in line or "FAIL:" in line or "ERROR:" in line:
                    match = re.search(r'(test_\w+|::test_\w+)', line)
                    if match:
                        failed.append(match.group(1))

        # Parse unittest output
        elif "======================================================================" in output:
            sections = output.split("======================================================================")
            for section in sections:
                if "FAIL:" in section or "ERROR:" in section:
                    match = re.search(r'(test_\w+)', section)
                    if match:
                        failed.append(match.group(1))

        return {
            "passed": passed,
            "failed": failed
        }

    @staticmethod
    def compare_test_results(before_output: str, after_output: str) -> Dict:
        """
        Compare test results before and after changes.

        Returns:
            dict with:
            - new_failures: tests that now fail (CRITICAL - regressions!)
            - new_passes: tests that now pass (good!)
            - still_failing: tests that failed before and still fail
            - still_passing: tests that passed before and still pass
        """
        before = EnhancedTestComparator.parse_test_output(before_output)
        after = EnhancedTestComparator.parse_test_output(after_output)

        before_passed = set(before['passed'])
        before_failed = set(before['failed'])
        after_passed = set(after['passed'])
        after_failed = set(after['failed'])

        return {
            "new_failures": list(before_passed & after_failed),  # REGRESSIONS!
            "new_passes": list(before_failed & after_passed),    # Good!
            "still_failing": list(before_failed & after_failed),
            "still_passing": list(before_passed & after_passed)
        }

    @staticmethod
    def extract_failure_details(output: str) -> List[str]:
        """
        Extract detailed failure information from test output.
        Returns list of failure descriptions with actual vs expected.
        """
        failures = []

        # Split by test failure sections
        sections = output.split("======================================================================")

        for section in sections:
            if "FAIL:" in section or "ERROR:" in section:
                failures.append(section.strip())

        return failures


# ============================================================================
# MULTI-SOLUTION CONSENSUS APPROACH
# ============================================================================

class MultiSolutionGenerator:
    """
    Generates multiple solutions and uses consensus/voting to select the best.
    Key benefits:
    - Reduces single-model errors
    - Finds more robust solutions
    - Can detect when models consistently produce same answer
    """

    @staticmethod
    def extract_solution_code(response: str) -> str:
        """
        Extract Python code from model response, handling markdown fences.
        """
        # Remove markdown code fences
        if "```python" in response:
            match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in response:
            match = re.search(r'```\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()

        return response.strip()

    @staticmethod
    def normalize_code(code: str) -> str:
        """
        Normalize code for comparison (remove comments, extra whitespace).
        """
        # Remove comments
        code = re.sub(r'#[^\n]*', '', code)
        # Remove empty lines
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    @staticmethod
    def find_consensus_solution(solutions: List[str]) -> Tuple[str, int, float]:
        """
        Find the most common solution among generated solutions.

        Args:
            solutions: List of solution strings

        Returns:
            tuple of (consensus_solution, count, confidence)
            confidence = count / total_solutions
        """
        if not solutions:
            return None, 0, 0.0

        # Normalize all solutions for comparison
        normalized = [MultiSolutionGenerator.normalize_code(s) for s in solutions]

        # Count occurrences
        counter = Counter(normalized)

        # Get most common
        most_common_normalized, count = counter.most_common(1)[0]

        # Find original solution that matches
        for i, norm in enumerate(normalized):
            if norm == most_common_normalized:
                consensus_solution = solutions[i]
                break

        confidence = count / len(solutions)

        logger.info(f"Consensus: {count}/{len(solutions)} solutions agree (confidence: {confidence:.1%})")

        return consensus_solution, count, confidence

    @staticmethod
    def analyze_solution_diversity(solutions: List[str]) -> Dict:
        """
        Analyze how diverse the generated solutions are.

        Returns:
            dict with unique_count, total_count, diversity_score, most_common_count
        """
        if not solutions:
            return {
                "unique_count": 0,
                "total_count": 0,
                "diversity_score": 0.0,
                "most_common_count": 0
            }

        normalized = [MultiSolutionGenerator.normalize_code(s) for s in solutions]
        counter = Counter(normalized)

        unique_count = len(counter)
        total_count = len(solutions)
        diversity_score = unique_count / total_count
        most_common_count = counter.most_common(1)[0][1]

        return {
            "unique_count": unique_count,
            "total_count": total_count,
            "diversity_score": diversity_score,
            "most_common_count": most_common_count
        }


# ============================================================================
# TEST RUNNER AUTO-DETECTION
# ============================================================================

class TestRunnerDetector:
    """
    Automatically detects test runner and mode from repository.
    Supports: pytest, unittest, custom test runners
    """

    @staticmethod
    def find_test_files() -> List[str]:
        """Find all test files in repository."""
        test_files = []
        for root, _, files in os.walk('.'):
            if '.git' in root:
                continue
            for file in files:
                if 'test_' in file and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        return sorted(test_files, key=len)

    @staticmethod
    def count_test_cases(file_path: str) -> int:
        """Count number of test functions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
            return len(test_functions)
        except:
            return 0

    @staticmethod
    def find_readme(start_path: str) -> Optional[str]:
        """Find README file by traversing up from given path."""
        current_dir = os.path.dirname(start_path)

        while current_dir and current_dir != '/':
            for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
                readme_path = os.path.join(current_dir, readme_name)
                if os.path.exists(readme_path):
                    return readme_path
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent

        return None

    @staticmethod
    def detect_from_readme(readme_path: str) -> Optional[str]:
        """Detect test runner from README content."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # Look for test runner mentions
            if 'pytest' in content:
                return 'pytest'
            elif 'unittest' in content and 'python -m unittest' in content:
                return 'python -m unittest'
            elif 'python test' in content:
                # Look for test runner file
                match = re.search(r'python\s+([\w/]+\.py)', content)
                if match:
                    return match.group(1)
        except:
            pass

        return None

    @staticmethod
    def detect_test_runner() -> Tuple[str, str]:
        """
        Detect test runner and mode.

        Returns:
            tuple of (test_runner, mode)
            mode is either "FILE" or "MODULE"
        """
        # Find test files
        test_files = TestRunnerDetector.find_test_files()

        if not test_files:
            logger.info("No test files found, defaulting to pytest")
            return "pytest", "FILE"

        # Find a good test file (with multiple tests)
        test_file = None
        for path in test_files:
            if TestRunnerDetector.count_test_cases(path) > 5:
                test_file = path
                break

        if not test_file:
            test_file = test_files[0] if test_files else None

        if not test_file:
            return "pytest", "FILE"

        # Check README
        readme_path = TestRunnerDetector.find_readme(test_file)
        if readme_path:
            runner = TestRunnerDetector.detect_from_readme(readme_path)
            if runner:
                logger.info(f"Detected test runner from README: {runner}")
                # Determine mode
                if runner == "pytest" or runner.endswith(".py"):
                    return runner, "FILE"
                else:
                    return runner, "MODULE"

        # Default to pytest
        logger.info("No test runner detected, defaulting to pytest")
        return "pytest", "FILE"


# ============================================================================
# ENHANCED SYNTAX VALIDATION
# ============================================================================

class EnhancedSyntaxValidator:
    """
    Enhanced syntax validation with detailed error reporting.
    """

    @staticmethod
    def validate_python_syntax(code: str, file_path: str = "<unknown>") -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax with detailed error reporting.

        Returns:
            tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code, filename=file_path)

            # Check for empty function bodies
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = list(node.body)
                    # Skip docstring
                    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                        body = body[1:]

                    # Check if empty or only pass
                    if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                        return False, f"Function '{node.name}' has empty body (line {node.lineno})"

            return True, None

        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            return False, error_msg
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def check_for_hardcoding(code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code contains hardcoded solutions (anti-cheating).

        Returns:
            tuple of (has_hardcoding, warning_message)
        """
        # Pattern: if input == specific_value: return specific_output
        hardcode_patterns = [
            r'if\s+\w+\s*==\s*\[.*?\]\s*:\s*return\s+\d+',
            r'if\s+\w+\s*==\s*\{.*?\}\s*:\s*return\s+\d+',
            r'if\s+\w+\s*==\s*["\'].*?["\']\s*:\s*return',
        ]

        for pattern in hardcode_patterns:
            if re.search(pattern, code):
                return True, "⚠️ Detected potential hardcoded solution. Use generic algorithms instead."

        return False, None
