from __future__ import annotations
import ast
import json
import os
import requests
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import csv
import logging
from uuid import uuid4
import urllib.request as _urlreq
# Removed: from concurrent.futures import ThreadPoolExecutor, as_completed (was used by oneshot embedding)
from collections import Counter
import math
import hashlib

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
        """Create a checkpoint (git commit + tag) at current state."""
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {"status": "error", "message": f"Not a git repository: {repo_path}"}

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if checkpoint already exists
            result = subprocess.run(["git", "tag", "-l", checkpoint_name], capture_output=True, text=True, check=False)
            if result.stdout.strip():
                os.chdir(original_dir)
                return {"status": "error", "message": f"Checkpoint '{checkpoint_name}' already exists"}

            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            # Check if there are changes to commit
            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)

            # Create commit if needed
            if status_result.stdout.strip():
                subprocess.run(["git", "commit", "-m", f"Checkpoint: {checkpoint_name}"], capture_output=True, text=True, check=True)

            # Get current commit hash
            hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit_hash = hash_result.stdout.strip()

            # Create tag
            subprocess.run(["git", "tag", checkpoint_name, commit_hash], check=True, capture_output=True)
            os.chdir(original_dir)

            return {"status": "success", "checkpoint_name": checkpoint_name, "commit_hash": commit_hash,
                    "message": f"Checkpoint '{checkpoint_name}' created at {commit_hash[:8]}"}

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"}
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

    @staticmethod
    def switch_checkpoint(repo_path: str, checkpoint_name: str, save_current: bool = True) -> dict:
        """Switch to a specific checkpoint, optionally stashing current changes."""
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {"status": "error", "message": f"Not a git repository: {repo_path}"}

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if checkpoint exists
            tag_result = subprocess.run(["git", "tag", "-l", checkpoint_name], capture_output=True, text=True, check=False)
            if not tag_result.stdout.strip():
                os.chdir(original_dir)
                return {"status": "error", "message": f"Checkpoint '{checkpoint_name}' not found"}

            # Save current state if requested
            stashed = False
            if save_current:
                status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
                if status_result.stdout.strip():
                    subprocess.run(["git", "stash", "push", "-u", "-m", f"Auto-stash before switching to {checkpoint_name}"],
                                   capture_output=True, text=True, check=True)
                    stashed = True

            # Checkout the checkpoint
            subprocess.run(["git", "checkout", checkpoint_name], capture_output=True, text=True, check=True)

            # Get commit hash
            hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit_hash = hash_result.stdout.strip()
            os.chdir(original_dir)

            return {"status": "success", "checkpoint_name": checkpoint_name, "commit_hash": commit_hash,
                    "message": f"Switched to checkpoint '{checkpoint_name}' at {commit_hash[:8]}", "stashed": stashed}

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"}
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

    @staticmethod
    def restore_stashed_changes(repo_path: str, stash_index: int = 0, remove_after_apply: bool = True) -> dict:
        """Restore previously stashed changes."""
        try:
            if not os.path.exists(os.path.join(repo_path, ".git")):
                return {"status": "error", "message": f"Not a git repository: {repo_path}"}

            original_dir = os.getcwd()
            os.chdir(repo_path)

            # Check if there are stashes
            stash_list_result = subprocess.run(["git", "stash", "list"], capture_output=True, text=True, check=True)
            if not stash_list_result.stdout.strip():
                os.chdir(original_dir)
                return {"status": "error", "message": "No stashed changes found"}

            stash_count = len(stash_list_result.stdout.strip().split('\n'))
            if stash_index >= stash_count:
                os.chdir(original_dir)
                return {"status": "error", "message": f"Stash index {stash_index} out of range. Only {stash_count} stash(es) available."}

            # Apply or pop the stash
            stash_ref = f"stash@{{{stash_index}}}"
            if remove_after_apply:
                command = ["git", "stash", "pop", stash_ref]
                action = "popped"
            else:
                command = ["git", "stash", "apply", stash_ref]
                action = "applied"

            subprocess.run(command, capture_output=True, text=True, check=True)
            os.chdir(original_dir)

            return {"status": "success", "message": f"Successfully {action} stash@{{{stash_index}}}",
                    "stash_index": stash_index, "removed": remove_after_apply}

        except subprocess.CalledProcessError as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Git stash command failed: {e.stderr if e.stderr else str(e)}"}
        except Exception as e:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}


# ============================================================================
# ENHANCED TEST COMPARISON
# ============================================================================

class EnhancedTestComparator:
    """Compares test results before and after changes to detect regressions."""

    @staticmethod
    def parse_test_output(output: str) -> Dict[str, List[str]]:
        """Parse test output to extract passed and failed tests."""
        passed = []
        failed = []

        # Parse pytest output
        if "PASSED" in output or "FAILED" in output:
            for line in output.split('\n'):
                if " PASSED" in line:
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

        return {"passed": passed, "failed": failed}

    @staticmethod
    def compare_test_results(before_output: str, after_output: str) -> Dict:
        """Compare test results before and after changes."""
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
        """Extract detailed failure information from test output."""
        failures = []
        sections = output.split("======================================================================")
        for section in sections:
            if "FAIL:" in section or "ERROR:" in section:
                failures.append(section.strip())
        return failures


# ============================================================================
# MULTI-SOLUTION CONSENSUS APPROACH
# ============================================================================

class MultiSolutionGenerator:
    """Generates multiple solutions and uses consensus/voting to select the best."""

    @staticmethod
    def extract_solution_code(response: str) -> str:
        """Extract Python code from model response, handling markdown fences."""
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
        """Normalize code for comparison (remove comments, extra whitespace)."""
        code = re.sub(r'#[^\n]*', '', code)
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    @staticmethod
    def find_consensus_solution(solutions: List[str]) -> Tuple[str, int, float]:
        """Find the most common solution among generated solutions."""
        if not solutions:
            return None, 0, 0.0

        normalized = [MultiSolutionGenerator.normalize_code(s) for s in solutions]
        counter = Counter(normalized)
        most_common_normalized, count = counter.most_common(1)[0]

        for i, norm in enumerate(normalized):
            if norm == most_common_normalized:
                consensus_solution = solutions[i]
                break

        confidence = count / len(solutions)
        logger.info(f"Consensus: {count}/{len(solutions)} solutions agree (confidence: {confidence:.1%})")
        return consensus_solution, count, confidence

    @staticmethod
    def analyze_solution_diversity(solutions: List[str]) -> Dict:
        """Analyze how diverse the generated solutions are."""
        if not solutions:
            return {"unique_count": 0, "total_count": 0, "diversity_score": 0.0, "most_common_count": 0}

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
    """Automatically detects test runner and mode from repository."""

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
            if 'pytest' in content:
                return 'pytest'
            elif 'unittest' in content and 'python -m unittest' in content:
                return 'python -m unittest'
            elif 'python test' in content:
                match = re.search(r'python\s+([\w/]+\.py)', content)
                if match:
                    return match.group(1)
        except:
            pass
        return None

    @staticmethod
    def detect_test_runner() -> Tuple[str, str]:
        """Detect test runner and mode."""
        test_files = TestRunnerDetector.find_test_files()
        if not test_files:
            logger.info("No test files found, defaulting to pytest")
            return "pytest", "FILE"

        test_file = None
        for path in test_files:
            if TestRunnerDetector.count_test_cases(path) > 5:
                test_file = path
                break

        if not test_file:
            test_file = test_files[0] if test_files else None

        if not test_file:
            return "pytest", "FILE"

        readme_path = TestRunnerDetector.find_readme(test_file)
        if readme_path:
            runner = TestRunnerDetector.detect_from_readme(readme_path)
            if runner:
                logger.info(f"Detected test runner from README: {runner}")
                if runner == "pytest" or runner.endswith(".py"):
                    return runner, "FILE"
                else:
                    return runner, "MODULE"

        logger.info("No test runner detected, defaulting to pytest")
        return "pytest", "FILE"


# ============================================================================
# ENHANCED SYNTAX VALIDATION
# ============================================================================

class EnhancedSyntaxValidator:
    """Enhanced syntax validation with detailed error reporting."""

    @staticmethod
    def validate_python_syntax(code: str, file_path: str = "<unknown>") -> Tuple[bool, Optional[str]]:
        """Validate Python syntax with detailed error reporting."""
        try:
            tree = ast.parse(code, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = list(node.body)
                    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                        body = body[1:]

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
        """Check if code contains hardcoded solutions (anti-cheating)."""
        hardcode_patterns = [
            r'if\s+\w+\s*==\s*\[.*?\]\s*:\s*return\s+\d+',
            r'if\s+\w+\s*==\s*\{.*?\}\s*:\s*return\s+\d+',
            r'if\s+\w+\s*==\s*["\'].*?["\']\s*:\s*return',
        ]

        for pattern in hardcode_patterns:
            if re.search(pattern, code):
                return True, "⚠️ Detected potential hardcoded solution. Use generic algorithms instead."

        return False, None


# ============================================================================
# 1. CENTRALIZED CONFIGURATION
# All settings, prompts, and constants are grouped here for easy management.
# ============================================================================
class AgentConfig:
    # Problem Types & Languages
    PROBLEM_TYPE_CREATE = "CREATE"
    PROBLEM_TYPE_FIX = "FIX"

    # Environment & Timeouts
    DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
    DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1800"))
    MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))
    MAX_FIX_TASK_STEPS = 400

    # Model Names
    GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
    KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
    DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
    QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]

    # Reasoning Effort Levels (temperature adjustments for different task complexities)
    REASONING_EFFORT = {
        "high": 0.0,      # Complex algorithm design, solution generation - CHANGED to 0.0 for deterministic output
        "medium": 0.0,    # Test generation, bug fixing, code analysis - CHANGED to 0.0 for deterministic output
        "low": 0.0,       # Simple tasks, code review, validation - CHANGED to 0.0 for deterministic output
    }

    # Context Window Management (based on model config.json max_position_embeddings)
    MODEL_CONTEXT_LIMITS = {
        GLM_MODEL_NAME: 131072,           # GLM-4.5-FP8: 131k tokens
        KIMI_MODEL_NAME: 131072,          # Kimi-K2-Instruct: 131k tokens
        DEEPSEEK_MODEL_NAME: 131072,      # DeepSeek-V3: 131k tokens
        QWEN_MODEL_NAME: 262144,          # Qwen3-Coder: 262k tokens (largest!)
    }
    DEFAULT_CONTEXT_LIMIT = 131072        # Default to 131k
    CONTEXT_SAFETY_MARGIN = 0.85          # Use 85% of limit to leave room for response
    MAX_OUTPUT_TOKENS = 4096              # Reserve for model output

    # Caching and State
    EMBED_CACHE_FILE = ".agent_embed_cache.json"
    STATE_FILE_PATH = ".agent_state.json"

    # Agent Behavior
    MAX_CONSECUTIVE_TOOL_FAILURES = 3

    # Prompts
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
    '''
    You are the problem type checker that will categories problem type into:

    1. CREATE: If the problem statement is about creating a new functionality from scratch.
    2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

    Only respond with the "FIX" or "CREATE".
    '''
    )

    ONESHOT_SYSTEM_PROMPT = """
    You are an autonomous programmer. The user will provide a bug report or 
    feature request (the "problem") plus a compact summary of the most 
    relevant repository files.  Your job is to return ONE *valid* unified 
    diff patch that fixes the problem. If you have any questions, do not ask the user. 
    Instead, solve it to the best of your ability with the knowledge you have.

    You will be provided with a summary of the repository files that are most relevant to the problem.
    Your patch must be valid and apply cleanly when run from the repository root.

    STRICT FORMAT RULES
    1. Return *only* the diff – no prose, no Markdown back-ticks.
    2. The diff must start with 'diff --git a/<path> b/<path>' followed by 
       the standard "--- a/<path>" and "+++ b/<path>" headers.
    3. Use -u style context hunks that begin with lines like @@ -N,M +N,M @@.
    4. Every changed file needs its own header block as in rule 2.
    5. End the patch with a trailing newline.

    Be exact: if the diff is syntactically malformed or wrapped in extra 
    text the automated patch tool will fail.

    OUTPUT RULES (VERY IMPORTANT, STRICT)
    • You MUST end your reply with a *raw* JSON object with "code_response" property – nothing else.
    • Must hold the unified diff from rules 1-5 *verbatim*.
    Example: {"code_response": "diff --git a/foo.py b/foo.py\\n..."}
    """

    DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent("""
    You're not allowed to repeat the same tool call with the same arguments.
    Your previous response: 
    {previous_response}

    Try to use something different!
    """)

    STUCK_DETECTION_PROMPT = textwrap.dedent("""
    🚨 **STUCK PATTERN DETECTED - NO PROGRESS** 🚨

    You've been repeating similar actions without making progress. This means your current approach is NOT working.

    **What to do NOW (5-Step Recovery):**
    1. **STOP** the current approach immediately
    2. **REFLECT**: Why isn't this working? What am I missing?
    3. **RE-READ** the problem statement carefully—did you misunderstand a requirement?
    4. **ANALYZE** test failures systematically:
       - What's the EXACT difference between actual vs expected?
       - Is it a TYPE issue (int vs float, object vs value)?
       - Is it a UNIT CONVERSION issue? (e.g., 145.6 vs 14560 = dollars vs cents!)
       - Is it an ERROR MESSAGE mismatch? (check problem statement for exact error text)
       - Is it a FORMAT issue (decimal places, list vs string)?
    5. **SWITCH STRATEGY** completely:
       - If searching in files A/B/C didn't work → try different files or read test code
       - If editing function X repeatedly → bug is likely in a DIFFERENT function
       - If tests show type mismatch → focus on output TYPE not algorithm logic
       - If tests show wrong value → re-check your algorithm/calculation
       - If error message mismatch → use EXACT error message text from problem statement
       - If tests pass then fail → you're missing part of the requirements

    **Common Stuck Patterns & Solutions:**
    - Same tool 3+ times → Try a DIFFERENT tool
    - Same search failing → Use BROADER or DIFFERENT keywords
    - Tests not improving → Read the TEST CODE itself, not just output
    - Same edit failing → The bug is in a DIFFERENT location

    **⚠️ CRITICAL WARNING - TEST EXPECTATIONS:**
    - If you think test expectations are "wrong" or "impossible" → YOU ARE LIKELY MISUNDERSTANDING THE PROBLEM
    - NEVER conclude that tests are wrong without first:
      1. Re-reading the problem statement 3 times
      2. Checking if you're testing against the RIGHT test file
      3. Verifying you understand the output type/units correctly
      4. Checking for unit conversions (cents vs dollars, etc.)
    - The test expectations are ALMOST ALWAYS correct - the issue is in YOUR understanding or implementation
    - **NEVER HARDCODE** specific test inputs/outputs - this is FORBIDDEN and will cause your solution to fail

    **DO NOT** repeat the same action. Try something COMPLETELY DIFFERENT.
    """)

    GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent("""
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
    
    ## 🧠 BEFORE CODING - ANALYZE THE PROBLEM:
    
    **⚠️ CRITICAL: Extract Examples & Determine Return Type:**
    1. **Find ALL examples** in problem statement (inputs → outputs)
    2. **Determine EXACT return type from examples:**
       • If examples show `8.00`, `51.20` → return **float** (dollars)
       • If examples show `800`, `5120` → return **int** (cents)
       • If examples show `"text"` → return **str**
       • If examples show `[1, 2, 3]` → return **list**
    3. **Check for units:**
       • Does problem mention "meters", "feet", "inches", "KM", "dollars" or "cents"?
       • Does problem show decimal points in examples?
       • Are there any conversion requirements?
    4. **VERIFY: Read the test code/examples CAREFULLY:**
       • What exact values are expected?
       • What type do the assertions use?
    
    **Identify Requirements:**
    - What calculations/logic are needed?
    - What edge cases must be handled?
    - What constraints exist?
    - What units are involved? (dollars vs cents, etc.)
    
    **⚠️ CRITICAL: Extract Error Messages from Problem Statement**
    1. **Find ALL error messages** mentioned in problem statement
    2. **Use EXACT text** - don't paraphrase or change wording
       • Example: If problem says "Tree could not be reoriented" → use EXACTLY that
       • Example: If problem says "No path found" → use EXACTLY that
    3. **Error Message Consistency Rules:**
       • If problem specifies one error for a class/type, use it for ALL similar cases
       • Unless problem explicitly says different messages for different methods
       • Example: If "ValueError('Invalid input')" shown once, use for all ValueError cases
    4. **Where to look for error messages:**
       • Problem statement examples
       • Error case descriptions
       • Expected behavior sections
    
    **Design Algorithm:**
    - What's the optimal approach?
    - How to handle each requirement?
    - Will this work for ALL cases, not just examples?
    
    **⚠️ VERIFY OUTPUT TYPE (MOST COMMON FAILURE POINT):**
    - **Read ALL examples** - what type do they show?
    - **Common mistake:** Confusing dollars vs cents, float vs int
    - **Examples:**
      • Problem shows `8.00`, test expects `8.00` → return `float` (8.00)
      • Problem shows `800`, test expects `800` → return `int` (800)
      • Problem shows `8`, test expects `8.00` → check test code to decide
    - **Units matter:**
      • If "price in dollars" + examples show decimals → return float dollars
      • If "price in cents" + examples show integers → return int cents
      • If unclear, prefer the type shown in examples
    - **Test your understanding:**
      • What does the problem statement explicitly say to return?
      • What format are the examples in?
      • Do test assertions use float or int?
    
    ## 💿 CRITICAL INSTRUCTIONS (Always Active):
    1. **Generic Solutions Only**: Write code that works for ANY input, not just the examples. No hardcoded values or problem-specific logic.
       ⚠️ **ABSOLUTELY FORBIDDEN**: Checking if input equals a specific value and returning a specific output (e.g., `if input_data == [1,2,3]: return 42`)
       ✅ **REQUIRED**: Implement the actual algorithm that computes the result for ANY input
    2. **Type Safety**: Pay attention to return types (int vs float, list vs string). Match the expected output type from examples.
    3. **Edge Cases**: Handle empty inputs, boundary values, None/null, and type variations generically.
    4. **Precision**: If examples show decimals (e.g., 51.2), return float. If examples show integers (e.g., 42), return int.
    5. **Units**: Pay attention to units (dollars vs cents, meters vs centimeters). Don't multiply/divide unnecessarily.
       ⚠️ **COMMON ERROR**: Returning dollars when tests expect cents (or vice versa) - check if you need to multiply/divide by 100!
    6. **Algorithm Correctness**: Think through your algorithm before coding. Verify logic is sound.

    Strict Requirements:
    1. Output the full content of Python files along with their file names.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.

    Return only the final python files code.

    Response Format Example:
    ```python
    file_1.py
    <complete file content>

    file_2.py
    <complete file content>
    ```
    """
    )


    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer with exceptional problem-solving skills. Your task is to generate a complete, working Python solution for the given problem statement.
    
    ## 🧠 INTELLIGENT PROBLEM ANALYSIS (Do this mentally first):
    
    **Step 1: Extract & Understand Examples**
    - Identify ALL example inputs and their expected outputs
    - Determine exact output TYPE from examples (int, float, string, list, etc.)
    - Note output FORMAT (decimal places, units, structure)
    - If examples show 12.5, output must be float; if examples show 12, output can be int
    
    **Step 2: Identify Constraints & Requirements**
    - What are the input constraints? (ranges, types, empty cases)
    - What are the business rules? (discounts, calculations, conditions)
    - What edge cases are hinted at? (empty inputs, single items, max values)
    - What are the output requirements? (units, precision, format)
    
    **Step 3: Detect Common Pitfalls**
    - Type mismatches (returning int when float expected)
    - Unit errors (cents vs dollars, multiply vs divide by 100)
    - Off-by-one errors (inclusive vs exclusive ranges)
    - Missing edge case handling (empty, None, zero)
    - Incorrect algorithm logic
    - **Error messages**: If problem mentions specific error messages or exceptions, use EXACTLY those words
    - **Error message consistency**: If problem specifies one error message for a class/module, use it consistently for ALL similar error cases unless explicitly told otherwise
    
    **Step 4: Design Algorithm**
    - What's the core algorithm needed? (DP, greedy, simulation, calculation)
    - What data structures are optimal?
    - How to handle each requirement systematically?
    
    **Step 5: Verify Logic Before Coding**
    - Mentally trace through examples with your algorithm
    - Will it produce the EXACT output shown? (type, value, format)
    - Are all edge cases covered?

    Strict Requirements:
    1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
    Return only the final Python code.

    Response Format Example:
    ```python
    file_1.py
    <complete file content>

    file_2.py
    <complete file content>
    ```
    """
    )

    GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
    You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

    ## 🚨 CRITICAL: Match Problem Statement EXACTLY
    
    **⚠️ Return Type Verification (CHECK THIS FIRST):**
    1. **Analyze problem examples** - what type do they show?
       • Examples: `8.00`, `51.20` → test with `assertEqual(func(), 8.00)` (float)
       • Examples: `800`, `5120` → test with `assertEqual(func(), 800)` (int)
       • Examples: `"text"` → test with `assertEqual(func(), "text")` (str)
    2. **Units matter:**
       • "Price in dollars" + decimal examples → expect float dollars
       • "Price in cents" + integer examples → expect int cents
    3. **In your tests, use the EXACT type from problem examples**
       • If problem shows `8.00`, write `self.assertEqual(total([1]), 8.00)` NOT `800`
       • If problem shows `800`, write `self.assertEqual(total([1]), 800)` NOT `8.00`
    
    **Error Messages & Exceptions:**
    - If problem statement specifies exact error messages (e.g., "raise ValueError('specific message')"), use EXACTLY those words
    - Don't invent your own error messages - extract them from problem statement
    - Pay attention to: ValueError messages, exception types, error text
    - **Use error messages CONSISTENTLY**: If problem specifies one error message for a class, use it for ALL error cases in that class unless problem explicitly says to use different messages for different methods
    
    **Example Precision:**
    - Match the EXACT output format from problem examples
    - If examples show specific exception messages, use those exact words in assertions
    - Don't assume - extract from problem statement
    - If problem shows an error message in one example, use that SAME message for all similar error scenarios

    Important things:
    1. Test functions declared in code skeleton, don't customized those prototypes.
    2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
    3. Do not generate testcases that are not mentioned in problem statement
    4. Minimize all testcases as you have context and generation limit
    5. **Extract exact error messages from problem statement** - don't make up your own

    Strict Requirements:
    1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).

    Response Format Example:
    ```python
    test_file_1.py
    <complete test file content>

    test_file_2.py
    <complete test file content>
    ```
    """
    )

    GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
    """
    You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

    Important things:
    1. Test functions declared in code skeleton, don't customized those prototypes.
    2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
    3. Do not generate testcases that are not mentioned in problem statement
    4. Minimize all testcases as you have context and generation limit

    Strict Requirements:
    1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).

    Response Format Example:
    ```python
    test_file_1.py
    <complete test file content>

    test_file_2.py
    <complete test file content>
    ```
    """
    )

    TESTCASES_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

    Important:
    1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
    2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
    3. Minimize all testcases as you have context and generation limit

    If no invalid testcases are detected and covered all critical edge cases:
    - Return the original code unchanged

    STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

    Response Format Example:
    ```python
    test_file_1.py
    <complete test file content>

    test_file_2.py
    <complete test file content>
    ```
    """
    )


    FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
    # Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
    
    ## 🧠 META-COGNITIVE AWARENESS (Self-Monitor Your Progress):
    
    **Track Your Progress:**
    - After every 3-4 actions, pause and reflect: Am I making progress toward the goal?
    - If tests are still failing after 5+ edits, your approach is likely wrong—rethink the strategy
    - If you've used the same tool 3 times with no progress, switch to a different approach
    - Keep a mental count: How many tests passing? Increasing or stuck?
    
    **Detect Stuck Patterns:**
    - Repeating the same search? You might be looking in the wrong place
    - Same edit keeps failing tests? The root cause is likely different than you think
    - Tests pass then fail again? You're missing the full scope of the problem
    
    **Adaptive Strategy Switching:**
    - If stuck: Step back, re-read problem statement, look for missed requirements
    - If tests fail: Don't guess—analyze the EXACT error, compare actual vs expected
    - If no progress: Try completely different approach (different files, different logic)
    
    ## 💿 CRITICAL INSTRUCTIONS (Always Active):
    These instructions are programmed to help you operate effectively. They apply to EVERY task:

    1. **Generic Solutions Only**: Never hardcode problem-specific logic. Your solution must work for ANY similar problem, not just this one.
       ⚠️ **ABSOLUTELY FORBIDDEN**: Checking if input equals a specific value and returning a hardcoded output (e.g., `if input_data == [1,2,3]: return 42`)
       ✅ **REQUIRED**: Fix the underlying algorithm to compute the result correctly for ALL inputs
       🚫 **Anti-Cheating System Active**: Hardcoding detection is enabled and will reject your changes if detected
    2. **No Assumptions**: Don't assume file structures, function names, or patterns. Always explore and verify.
    3. **Edge Cases**: Always consider and handle edge cases generically (empty inputs, boundary values, null/None, type variations).
    4. **Type Safety**: Ensure return types match expectations (int vs float, string vs list, etc.).
       ⚠️ **COMMON ERROR**: Check for unit conversion issues (e.g., 145.6 vs 14560 = dollars vs cents!)
    5. **Algorithm Correctness**: Verify your logic is sound before implementing. Think through the algorithm step-by-step.
    6. **No Shortcuts**: Don't skip validation steps. Test thoroughly before calling finish().
    7. **Learn from Failures**: If a tool fails twice with the same approach, switch strategies immediately.
    8. **Precision Matters**: Pay attention to output formats, decimal places, units (dollars vs cents), and data types.
    
    ## 🎯 INTELLIGENT PROBLEM-SOLVING WORKFLOW:
    
    **Phase 1: Deep Understanding (ALWAYS DO FIRST)**
    1. Read the problem statement CAREFULLY—extract:
       - What is the EXACT issue? (bug, feature, performance)
       - What are the symptoms? (error message, wrong output, crash)
       - Are there examples showing expected behavior?
       - What constraints or requirements are mentioned?
    
    **Phase 2: Strategic Investigation**
    2. Locate relevant files intelligently:
       - Start with error traces (if provided)
       - Search for keywords from the problem
       - Read test files to understand expected behavior
       - Don't randomly explore—be targeted
    
    **Phase 3: Root Cause Analysis**
    3. Before editing, understand WHY the issue exists:
       - Read the problematic code carefully
       - Trace through the logic mentally
       - Compare with expected behavior
       - Identify the EXACT line/logic causing the issue
    
    **Phase 4: Precise Implementation**
    4. Make minimal, targeted changes:
       - Fix the root cause, not symptoms
       - Handle edge cases the original code missed
       - Ensure types/formats match requirements
       - Maintain backward compatibility
    
    **Phase 5: Comprehensive Validation (Test-Driven Debugging)**
    5. Test thoroughly with systematic analysis:
       - Run tests after EVERY significant change
       - **When tests fail, SYSTEMATIC analysis (follow this checklist):**
         1. **Which tests failed?** List them
         2. **Error type?** AssertionError, TypeError, AttributeError, etc.
         3. **Expected vs Actual?** Extract EXACT values
         4. **⚠️ TYPE MISMATCH? (MOST COMMON)**
            • Is actual `145.6` (float) but expected `14560` (int)? → Unit conversion issue (dollars vs cents)
            • Is actual `8` (int) but expected `8.0` (float)? → Return type wrong
            • Is actual `"[1,2]"` (str) but expected `[1,2]` (list)? → Type conversion needed
         5. **ERROR MESSAGE mismatch?** Check problem for exact error text
         6. **FORMAT issue?** Decimal places, list structure, string format
         7. **LOGIC error?** Algorithm incorrect for this case
       - **Read the failing TEST CODE** (don't just read output):
         • What EXACTLY is the test checking?
         • What type/format does it expect?
         • What error message does it expect?
       - Fix root cause based on analysis, not random guesses
    
    **Phase 6: Adaptive Recovery**
    6. If tests keep failing:
       - Re-read problem—did you miss something?
       - Check test code—what EXACTLY is it expecting?
       - Compare your output vs expected—spot the pattern
       - If same error after 3-4 attempts, SIMPLIFY your approach
       - If getting recursion errors, check for infinite loops/missing base cases
       - Try completely different approach if stuck

    ## ⚠️ CRITICAL: Complexity Management
    - If your solution is getting complex (many nested functions, deep recursion), SIMPLIFY
    - Prefer iterative solutions over deep recursion when possible
    - Always include base cases to prevent infinite recursion
    - Test simple cases first before complex ones
    
    ## ⏱️ STEP MANAGEMENT & EFFICIENCY:
    
    **You have limited steps - use them wisely:**
    - **Efficiency Guidelines:**
      • Don't repeat same action > 2 times without progress
      • If stuck after 3 attempts → change strategy completely
      • Run tests after EVERY significant code change
      • Finish when ALL tests pass (don't over-iterate)
    
    - **When to call finish():**
      ✅ All tests passing
      ✅ Solution works correctly
      ✅ No obvious bugs remaining
      ❌ DON'T wait for "perfect" code
      ❌ DON'T keep testing if already passing
    
    - **Step Efficiency Red Flags:**
      • Same tool used 5+ times → You're stuck, change approach
      • Same file read 3+ times → Inefficient, take notes
      • Tests run without changes → Wasting steps
      • No progress in 5 steps → Wrong approach, pivot

    ## Follow these steps to fix the issue:
    1. As a first step, find the relevant files in the repo to work on.
    2. Localise the code causing the issue.
    3. Edit the sourcecode of the repo to resolve the issue.
    4. Think about edgecases and make sure the fix handles them as well.
    5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
    6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
    7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
    8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
    9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
    10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
    11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
    12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
    13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.

    ## 🚨 TOOL USAGE RULES (CRITICAL):
    - **search_in_all_files_content**: Use ONLY for searching CODE CONTENT, NEVER for file/directory paths or import paths
    - **Finding Files/Directories**: 
      * First, search for the file/class/function name in all files using search_in_all_files_content
      * Use directory tree exploration if available
      * Infer locations from import statements in existing files
      * DO NOT assume specific files exist - always verify first
    - **Repetition Detection**: Stop after 2 "not found" results, switch strategy after 2 identical tool failures
    - **Regex in Search**: Escape parentheses `()` with `\\(\\)`

    ## Multi-file awareness (critical):
    - Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
    - Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
    - Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
    - Re-run tests only after covering all discovered occurrences to avoid partial fixes.

    ## Test generation guidance:
    - Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.
    - Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.
    - Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
    - Keep generated tests minimal and focused on the bug and its edge cases.
    - Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.

    You have access to the following tools:-
    {tools_docs}

    {format_prompt}
    """)

    FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
    # Now let's start. Here is the problem statement:
    {problem_statement}
    
    ## 🚨 CRITICAL REMINDER: If the problem statement above specifies exact error messages (e.g., "raise ValueError('specific text')"), you MUST use those EXACT words in your implementation. Do not invent your own error messages.
    """)


    FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that can find the test runner for a given repository.
    - The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
    - Do not use the test runner to run test for whole repository or test setup.
    - Read the README file and find the test runner. If there is no test runner, return pytest.
    - Output format should be as the following. No other texts are allowed.
    abc/test.py
    """)

    TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that determines the mode of the test runner.
    Read the test runner file and determine if it requires a module or a file path to run the test.
    Output should be one of MODULE or FILE, No other texts are allowed.
    - MODULE: When the test runner requires a module path to run the test.
    - FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
    """)


    STOP_INSTRUCTION=textwrap.dedent("""
    # 🎨 
    DO NOT generate `observation:` in your response. It will be provided by user for you.
    Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
    """)

    FORMAT_PROMPT_V0=textwrap.dedent("""
    **📝 Response Format Requirements**

    1. **Strict Triplet Format**:
       - `next_thought`: Detailed reasoning (include:
         - Problem understanding
         - Code analysis
         - Solution justification
         - Validation plan)
       - `next_tool_name`: Must be an exact tool name from the tool list
       - `next_tool_args`: Valid JSON with:
         - Proper escaping
         - No trailing commas
         - Tool-specific parameters

    2. **Error Handling Format**:
       - For errors: 
         next_thought: "Error: [detailed explanation]"
         next_tool_name: ""
         next_tool_args: {}

    3. **Example Valid Format**:
       next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
       next_tool_name: "apply_code_edit"
       next_tool_args: {
         "file_path": "network.py",
         "search": "return json.loads(response)",
         "replace": "try:\\n    return json.loads(response)\\nexcept JSONDecodeError:\\n    logger.error(f'Invalid JSON: {{response}}')\\n    raise"
       }

    4. **Invalid Format Examples** (Avoid These):
       - Missing any of the three required fields
       - JSON syntax errors in next_tool_args
       - Extra text outside the triplet format
       - Using incorrect tool names
       - Not quoting special characters properly
    """)

# ============================================================================
# Global Variables & Setup
# ============================================================================
REPO_DIR = ""
DEBUG_MODE = True
RUN_ID = os.getenv("RUN_ID", "")
run_id = RUN_ID  # Legacy compatibility

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear existing handlers
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# ============================================================================
# 2. STATE MANAGEMENT & CHAIN OF THOUGHT (with recovery)
# ============================================================================
class EnhancedCOT:
    class Action:
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str, is_error:bool=False, raw_response:str=None, total_attempts:int=0, inference_error_counter:dict=None, request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False

        def to_dict(self):
            return {
                "next_thought": self.next_thought,
                "next_tool_name": self.next_tool_name,
                "next_tool_args": self.next_tool_args,
                "observation": self.observation,
                "is_error": self.is_error,
                "raw_response": self.raw_response,
                "total_attempts": self.total_attempts,
                "inference_error_counter": self.inference_error_counter,
                "request_data": self.request_data,
                "is_deleted": self.is_deleted,
            }

        @classmethod
        def from_dict(cls, data: dict):
            # Compatibility for loading state saved without these keys
            data.setdefault("total_attempts", 0)
            data.setdefault("inference_error_counter", None)
            data.setdefault("request_data", None)
            # Remove keys that are not in the constructor
            valid_keys = inspect.signature(cls.__init__).parameters.keys()
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered_data)

    def __init__(self, latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep

    def add_action(self, action: EnhancedCOT.Action):
        self.thoughts.append(action)

    def is_thought_repeated(self)->bool:
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False

    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            assistant_str = (
                f"next_thought:{thought.next_thought}\n"
                f"next_tool_name:{thought.next_tool_name}\n"
                f"next_tool_args:{json.dumps(thought.next_tool_args)}\n"
            )
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                if thought.observation is None: _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)): _obs_len = len(thought.observation)
                else: _obs_len = len(str(thought.observation).splitlines())
                user_str = (f"observation: {'error occurred.' if thought.is_error else ''} "
                            f"output omitted ({_obs_len}) lines\n")
            else:
                try:
                    obs_render = json.dumps(list(thought.observation), ensure_ascii=False) if isinstance(thought.observation, (list, tuple)) else str(thought.observation)
                except (TypeError, OverflowError):
                    obs_render = str(thought.observation)
                user_str = f"observation: {obs_render}"

            messages.append({"role": "assistant", "content": assistant_str})
            messages.append({"role": "user", "content": user_str})
        return messages

    def save_state(self, filepath: str):
        """Saves the current state of the COT to a JSON file."""
        try:
            state = {
                "thoughts": [action.to_dict() for action in self.thoughts],
                "latest_observations_to_keep": self.latest_observations_to_keep,
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Successfully saved agent state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")

    @classmethod
    def load_state(cls, filepath: str):
        """Loads agent state from a JSON file."""
        if not os.path.exists(filepath):
            logger.info(f"State file not found at {filepath}. Starting with a fresh state.")
            return cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            cot = cls(state.get("latest_observations_to_keep", 5))
            cot.thoughts = [cls.Action.from_dict(action_data) for action_data in state.get("thoughts", [])]
            logger.info(f"Successfully loaded agent state from {filepath} with {len(cot.thoughts)} actions.")
            return cot
        except Exception as e:
            logger.error(f"Failed to load agent state from {filepath}, starting fresh. Error: {e}")
            return cls()

    def export_to_csv(self,file_path:str="./xray.csv"):
        with open(file_path, "w", newline='') as f:
            writer=csv.writer(f)
            writer.writerow(["next_thought","next_tool_name","next_tool_args","observation","is_error","raw_response","total_attempts", "inference_error_counter", "request_data", "request_data_len", "is_deleted"])
            if len(self.thoughts)>0:
                for thought in self.thoughts:
                    writer.writerow([
                        thought.next_thought, thought.next_tool_name, json.dumps(thought.next_tool_args), 
                        thought.observation, thought.is_error, thought.raw_response, thought.total_attempts,
                        str(thought.inference_error_counter), str(thought.request_data), 
                        len(str(thought.request_data)), thought.is_deleted
                    ])

# ============================================================================
# Utilities, Parsers, and Network Layer
# ============================================================================
class Utils:
    @classmethod
    def get_available_modules(cls) -> set[str]:
        import sys, pkgutil
        available: set[str] = set(sys.builtin_module_names)
        for module_info in pkgutil.iter_modules():
            top_level = module_info.name.split(".")[0]
            available.add(top_level)
        return available

    @classmethod
    def message_to_str(cls,messages:list[dict]): 
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages)
    
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        return strings
        
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception:
            try:
                # Be cautious with eval, but it's in the original code.
                # A safer alternative would be ast.literal_eval
                return ast.literal_eval(json_string)
            except Exception:
                logger.info(f"Unable to fix JSON manually, trying with LLM.")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                raise JSONDecodeError(f"Invalid JSON after attempting to fix: {json_string}", json_string, 0)
    
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        try:
            with open("../failed_messages.csv","a", newline='') as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])
        except IOError as e:
            logger.error(f"Could not write to failed_messages.csv: {e}")


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        start_line_num = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        end_line_num = getattr(node, 'end_lineno', start_line_num)
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[start_line_num-1:end_line_num])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": start_line_num
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node): self._process_function(node)
    def visit_AsyncFunctionDef(self, node): self._process_function(node)
    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        start_line_num = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        end_line_num = getattr(node, 'end_lineno', start_line_num)
        lines = self.file_content.split("\n")
        body = "\n".join(lines[start_line_num-1:end_line_num])
        self.classes[node.name] = {"body": body, "line_number": start_line_num}
        self.generic_visit(node)

class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE, RESERVED_TOKEN_PRESENT, RATE_LIMIT_EXCEEDED, INVALID_RESPONSE_FORMAT, TIMEOUT, UNKNOWN, NETWORK_ERROR, AUTHENTICATION_ERROR, RESOURCE_EXHAUSTED = range(9)
    
    # Circuit breaker: track model health
    _model_failure_counts = {}
    _model_circuit_open = {}
    _CIRCUIT_THRESHOLD = 3  # Open circuit after 3 consecutive failures
    _CIRCUIT_RESET_TIME = 300  # Reset after 5 minutes
    
    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str | None]:
        if isinstance(raw_text, dict) and raw_text.get("error"): return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not isinstance(raw_text, str): raw_text = str(raw_text)
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"): return False, "Incomplete response, must end with '}' or '}]'"
        if not raw_text: return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text: return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text: return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text: return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text: return False, cls.ErrorType.NETWORK_ERROR.name
        if '500 Server Error' in raw_text or 'Internal Server Error' in raw_text: return False, "SERVER_ERROR"
        return True, None
    
    @classmethod
    def is_circuit_open(cls, model: str) -> bool:
        """Check if circuit breaker is open for a model."""
        if model not in cls._model_circuit_open:
            return False
        
        open_time = cls._model_circuit_open[model]
        if time.time() - open_time > cls._CIRCUIT_RESET_TIME:
            # Reset circuit after timeout
            logger.info(f"[CIRCUIT] Resetting circuit for {model} after {cls._CIRCUIT_RESET_TIME}s")
            del cls._model_circuit_open[model]
            cls._model_failure_counts[model] = 0
            return False
        
        return True
    
    @classmethod
    def record_model_result(cls, model: str, success: bool):
        """Track model success/failure for circuit breaker."""
        if model not in cls._model_failure_counts:
            cls._model_failure_counts[model] = 0
        
        if success:
            cls._model_failure_counts[model] = 0  # Reset on success
            if model in cls._model_circuit_open:
                logger.info(f"[CIRCUIT] Model {model} recovered, closing circuit")
                del cls._model_circuit_open[model]
        else:
            cls._model_failure_counts[model] += 1
            if cls._model_failure_counts[model] >= cls._CIRCUIT_THRESHOLD:
                if model not in cls._model_circuit_open:
                    logger.warning(f"[CIRCUIT] Opening circuit for {model} after {cls._CIRCUIT_THRESHOLD} failures")
                    cls._model_circuit_open[model] = time.time()

    @classmethod
    def get_error_counter(cls)->dict[str,int]: return {k.name: 0 for k in cls.ErrorType}

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user. Reply only with the valid JSON string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response = cls.make_request(messages, model=AgentConfig.DEEPSEEK_MODEL_NAME)
        try:
            return json.loads(response.strip().strip('```json').strip('```'))
        except JSONDecodeError:
            logger.error(f"Error fixing json string after LLM attempt. LLM response: {response}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,temperature:float=0.0)->str:
        global run_id
        
        # Check circuit breaker - skip if circuit is open
        if cls.is_circuit_open(model):
            logger.warning(f"[CIRCUIT] Circuit open for {model}, trying fallback model")
            # Try to find a healthy model
            for fallback_model in AgentConfig.AGENT_MODELS:
                if not cls.is_circuit_open(fallback_model):
                    logger.info(f"[CIRCUIT] Using fallback model: {fallback_model}")
                    model = fallback_model
                    break
            else:
                # All circuits open, use original model anyway
                logger.warning(f"[CIRCUIT] All circuits open, using {model} anyway")
        
        url = f"{AgentConfig.DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        logger.debug(f"[REQUEST] run_id: {run_id}, model: {model}")

        # Apply context window management
        messages = truncate_messages_to_fit(messages, model, preserve_system=True)

        request_data = {"run_id": run_id or "1", "messages": messages, "temperature": temperature, "model": model}
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=request_data, timeout=200, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            is_oai_interface = isinstance(response_json, dict) and response_json.get('choices')
            if is_oai_interface:
                result = response_json['choices'][0]['message']['content'].lstrip()
                cls.record_model_result(model, success=True)
                return result
            
            raw_text = str(response_json).strip()
            if isinstance(raw_text, dict):
                cls.record_model_result(model, success=True)
                return str(raw_text) # Should already be handled but as a fallback
            cls.record_model_result(model, success=True)
            return raw_text.lstrip()
            
        except requests.exceptions.Timeout:
            cls.record_model_result(model, success=False)
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.RequestException as e:
            cls.record_model_result(model, success=False)
            return f"ERROR: Request failed for model {model}: {e}"
        except JSONDecodeError:
            cls.record_model_result(model, success=False)
            return f"ERROR: Invalid JSON response for model {model}"
        except (KeyError, IndexError, TypeError) as e:
            cls.record_model_result(model, success=False)
            return f"ERROR: Invalid response structure for model {model}: {e}"
        except Exception as e:
            cls.record_model_result(model, success=False)
            return f"ERROR: Unexpected error for model {model}: {e}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: list, model: str, max_retries: int = 5, base_delay: float = 1.0, temperature: float = 0.0):
        error_counter = cls.get_error_counter()
        total_attempts = 0
        original_model_index = AgentConfig.AGENT_MODELS.index(model) if model in AgentConfig.AGENT_MODELS else -1

        for attempt in range(max_retries):
            total_attempts += 1
            if original_model_index != -1:
                # Skip models with open circuit breaker
                attempts_to_find_healthy_model = 0
                while attempts_to_find_healthy_model < len(AgentConfig.AGENT_MODELS):
                    current_model = AgentConfig.AGENT_MODELS[(original_model_index + attempt + attempts_to_find_healthy_model) % len(AgentConfig.AGENT_MODELS)]
                    if not cls.is_circuit_open(current_model):
                        break
                    logger.info(f"[CIRCUIT] Skipping {current_model} (circuit open)")
                    attempts_to_find_healthy_model += 1
                else:
                    # All circuits open, use original model anyway
                    logger.warning(f"[CIRCUIT] All models have open circuits, using {model} anyway")
                    current_model = model
            else:
                current_model = model # If not in list, just keep retrying the same one
            
            raw_text = cls.make_request(messages, model=current_model, temperature=temperature)
            
            is_valid, error_msg = cls.is_valid_response(raw_text)
            if not is_valid:
                error_body = error_msg or "Unknown validation error"
                cls.record_model_result(current_model, success=False)
            else:
                next_thought, next_tool_name, next_tool_args, error_msg_parse = cls.parse_response(raw_text)
                if not error_msg_parse:
                    cls.record_model_result(current_model, success=True)
                    return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages
                error_body = error_msg_parse
                cls.record_model_result(current_model, success=False)

            logger.error(f"Attempt {attempt + 1}/{max_retries} failed with model {current_model}. Error: {error_body}. Response: {raw_text[:200]}...")
            error_name = next((e.name for e in cls.ErrorType if e.name in error_body.upper()), "UNKNOWN")
            if "SERVER_ERROR" in error_body:
                error_name = "SERVER_ERROR"
            error_counter[error_name] = error_counter.get(error_name, 0) + 1
            
            if attempt < max_retries - 1:
                if error_name not in ["RATE_LIMIT_EXCEEDED", "TIMEOUT", "EMPTY_RESPONSE", "SERVER_ERROR"]:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({"role": "user", "content": f"observation: {error_body}"})
                time.sleep(random.uniform(1.2 * base_delay, 1.5 * base_delay))
            else:
                raise RuntimeError(f"Failed after {max_retries} retries. Last error: {error_body}")
    
    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern_parts = [f'"{re.escape(k)}":\\s*(.*)' for k in arguments]
        pattern = r',\s*'.join(pattern_parts)
        match = re.search(pattern, json_string, re.DOTALL)

        if not match: return f"Error: Cannot match pattern for args {arguments} in string {json_string}"
        
        result_json = {}
        for i, arg in enumerate(arguments):
            value = match.group(i+1).strip()
            # Clean up value that might be part of the next key-value pair
            if i < len(arguments) - 1:
                next_key = f'"{arguments[i+1]}"'
                if next_key in value:
                    value = value.split(next_key)[0].strip().rstrip(',')
            
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            value = value.replace('\\n', '\n')
            result_json[arg] = value
        return result_json

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args_str: str) -> dict:
        clean_str = next_tool_args_str.strip().strip('```json').strip('```')
        try:
            return Utils.load_json(clean_str)
        except (JSONDecodeError, SyntaxError, ValueError) as e:
            logger.warning(f"Standard JSON parsing failed for tool '{tool_name}': {e}. Attempting manual parse.")
            required_args = EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True)
            if not required_args: # Tool might have no required args
                return {}
            parsed = cls.parse_malformed_json(required_args, clean_str)
            if isinstance(parsed, str):
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_RESPONSE_FORMAT, parsed)
            return parsed

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = "1", temperature: float = 0.0):
        cleaned_msgs = [m for m in messages if m.get("role") in {"system", "user", "assistant", "tool"} and m.get("content", "").strip()]
        if not cleaned_msgs: raise RuntimeError("No valid messages to send.")
        
        return cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
    
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"['\"]*next_thought['\"]*\s*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*\s*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*\s*:", "next_tool_args:", text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp:
            text_resp = "next_thought: " + text_resp
        return text_resp

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any, str | None]:
        text_resp = text_resp.strip().split("observation:")[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)

        thought_match = re.search(r"next_thought:(.*?)(?=next_tool_name:)", text_resp, re.DOTALL)
        name_match = re.search(r"next_tool_name:(.*?)(?=next_tool_args:)", text_resp, re.DOTALL)
        args_match = re.search(r"next_tool_args:(.*)", text_resp, re.DOTALL)

        if not (thought_match and name_match and args_match):
            error_msg = "Invalid response format: could not find all required fields (next_thought, next_tool_name, next_tool_args)."
            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg

        next_thought = thought_match.group(1).strip()
        next_tool_name_raw = name_match.group(1).strip()
        next_tool_args_raw = args_match.group(1).strip()
        
        try:
            next_tool_name = next_tool_name_raw.strip("'\"")
            next_tool_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
            return next_thought, next_tool_name, next_tool_args, None
        except Exception as e:
            error_msg = f"Failed to parse tool arguments: {e}"
            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg

# ============================================================================
# TOOL MANAGER FRAMEWORK AND IMPLEMENTATIONS
# ============================================================================

class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR, RUNTIME_ERROR, TIMEOUT, FILE_NOT_FOUND, SEARCH_TERM_NOT_FOUND, UNKNOWN, THIRD_PARTY_DEPENDENCIES, MULTIPLE_SEARCH_RESULTS_FOUND, BUG_REPORT_REQUIRED, INVALID_RESPONSE_FORMAT, INVALID_TOOL_NAME, INVALID_FILE_PATH, INVALID_TOOL_CALL, IMPORT_ERROR, GIT_OPERATION_FAILED, GIT_CONFIG_ERROR, GIT_STATE_ERROR, GIT_MERGE_CONFLICT, GIT_BRANCH_ERROR, TEST_COVERAGE_ERROR, DEPENDENCY_ANALYSIS_ERROR, CODE_SMELL_DETECTION_ERROR, GIT_HISTORY_ERROR, CODE_QUALITY_ERROR, SOLUTION_VALIDATION_ERROR, CODE_STYLE_ERROR, SOLUTION_COMPARISON_ERROR = range(27)
            
        def __init__(self,error_type,message:str):    
            # Handle both enum instances and strings for backward compatibility
            if isinstance(error_type, str):
                self.error_type = error_type
            else:
                self.error_type = error_type.name
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True
        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            doc += "\n\nOutput: " + output_description[1].strip()
        
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self': continue
            if param.default is param.empty: required.append(param.name)
            
            param_description_match = re.search(f"{param.name}:([^\n]+)", doc_fn)
            param_description = param_description_match.group(1).strip() if param_description_match else f"Parameter '{param.name}'"
            
            type_hint = str(param.annotation)
            if "list" in type_hint.lower() and "str" in type_hint:
                properties[param.name] = {"type": "array", "items": {"type": "string"}, "description": param_description}
            elif 'str' in type_hint: properties[param.name] = {"type": "string", "description": param_description}
            elif 'int' in type_hint: properties[param.name] = {"type": "integer", "description": param_description}
            elif 'float' in type_hint: properties[param.name] = {"type": "number", "description": param_description}
            elif 'bool' in type_hint: properties[param.name] = {"type": "boolean", "description": param_description}
            else: properties[param.name] = {"type": "string", "description": param_description}
        
        return {
            "name": name,
            "description": doc.strip(),
            "input_schema": {"type": "object", "properties": properties, "required": required}
        }

    @classmethod
    def get_tool_args_for_tool(cls, tool_name:str, required:bool=False)->list[str]:
        if tool_name not in cls.TOOL_LIST:
            # Return empty list for unknown tools to prevent crash during parsing
            return []
        schema = cls.TOOL_LIST[tool_name]['input_schema']
        if required:
            return schema.get('required', [])
        return list(schema['properties'].keys())

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_NAME, f"Error: tool '{tool_name}' not found.")
        tool_method = getattr(self, tool_name, None)
        if not (tool_method and callable(tool_method)):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_NAME, f"Error: tool '{tool_name}' is not a callable method.")
        return tool_method
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR,f"Syntax error. {str(e)}")
    
    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> tuple[bool, str]:
        """Public wrapper for syntax checking. Returns (is_error: bool, error_msg: str)."""
        return self._check_syntax_error(content, file_path)

    def _detect_hardcoding(self, file_path: str, content: str) -> tuple[bool, str]:
        """
        Detects hardcoded solutions that check for specific inputs and return specific outputs.
        This prevents agents from "cheating" by hardcoding test cases instead of solving generally.

        Returns:
            (is_hardcoded: bool, warning_message: str)
        """
        # Skip test files - they legitimately contain specific test data
        if 'test' in file_path.lower() or file_path.startswith('.'):
            return False, ""

        import re

        # Pattern 1: Exact equality checks against complex data structures
        # Examples: if data == [1,2,3,4]: return 100
        #           if input == "specific string": return "result"
        patterns_complex = [
            r'if\s+\w+\s*==\s*\[.*?\]\s*:',  # if var == [list]:
            r'if\s+\w+\s*==\s*\(.*?\)\s*:',  # if var == (tuple):
            r'if\s+\w+\s*==\s*\{.*?\}\s*:',  # if var == {dict}:
        ]

        for pattern in patterns_complex:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                # Check if there are multiple such patterns (strong indicator of hardcoding)
                if len(matches) > 1:
                    return True, f"Detected {len(matches)} hardcoded input checks in {file_path}:\n{matches[:3]}\n\nThis suggests hardcoding specific test cases instead of implementing a general algorithm."

                # Even one case might be hardcoding - check the context
                for match in matches:
                    # If the comparison involves a complex structure (length > 20 chars), likely hardcoded
                    if len(match) > 40:
                        return True, f"Detected hardcoded input check with complex data structure in {file_path}:\n{match}\n\nThis appears to be hardcoding a specific test case. Implement a general solution instead."

        # Pattern 2: Multiple return statements with hardcoded numeric values after input checks
        # This catches patterns like:
        #   if data == [...]: return 68.00
        #   if data == [...]: return 114.00
        if_return_pattern = r'if\s+\w+\s*==.*?:\s*return\s+[\d.]+'
        if_returns = re.findall(if_return_pattern, content, re.DOTALL)
        if len(if_returns) >= 2:
            return True, f"Detected multiple ({len(if_returns)}) hardcoded input-output mappings in {file_path}.\n\nThis is a clear sign of hardcoding test cases instead of solving the problem generally."

        return False, ""

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            # Check for hardcoded solutions (anti-cheating detection)
            hardcoding_detected, hardcoding_warning = self._detect_hardcoding(file_path, content)
            if hardcoding_detected:
                logger.warning(f"[ANTI-HARDCODE] Potential hardcoding detected in {file_path}")
                logger.warning(f"[ANTI-HARDCODE] {hardcoding_warning}")
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                    f"❌ HARDCODING DETECTED - This violates the 'Generic Solutions Only' principle!\n\n{hardcoding_warning}\n\n🚫 Your solution must work for ANY input, not just specific test cases.\n💡 Instead of hardcoding specific inputs, fix the underlying algorithm."
                )

            with open(file_path, "w") as file:
                file.write(content)
            return f"File {file_path} saved successfully"
        else:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR, "Error saving file. " + error.message)

    def _run_code(self,content:str,file_path:str)->str:
        self._save(file_path, content)
        # Simplified dependency check for robustness
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr: error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr: error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type, f"Error running code: {result.stderr}\n")
        return f"{result.stdout}\n"
    
    def get_final_git_patch(self) -> str:
        try:
            command = """
            shopt -s globstar
            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore
            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true
            git diff --cached > .patch.txt
            cat .patch.txt
            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            logger.info("Generating final git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"


class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(self, available_tools: Optional[list[str]] = None, test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]

        # Clear and rebuild TOOL_LIST for this instance
        self.__class__.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.__class__.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: continue
                    self.__class__.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={k:{e.name:0 for e in self.Error.ErrorType} for k in self.__class__.TOOL_LIST.keys()}
        self.tool_invocations={k:0 for k in self.__class__.TOOL_LIST.keys()}

    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term:
            logger.debug(f"Search term specified: '{search_term}', using search_in_specified_file_v2")
            return self.search_in_specified_file_v2(file_path, search_term)
        
        # Check if file exists before trying to open
        if not os.path.exists(file_path):
            # Provide helpful error message with suggestions
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND,
                f"File '{file_path}' does not exist. Try using search_in_all_files_content to find the correct file path first."
            )
        
        with open(file_path, "r") as f:
            if search_start_line or search_end_line:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()
        return Utils.limit_strings(content, n=limit) if limit != -1 else content
    
    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL, "Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        logger.info(f"Solutions proposed. Selected index: {selected_solution}. Reason: {reason_for_selection}")
        
        # Enhanced validation with clearer error messages
        if not isinstance(solutions, list):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL, 
                f"Error: 'solutions' must be a list, got {type(solutions).__name__}.\n"
                "Please provide a list of at least 2 solution descriptions.\n"
                "Example: solutions=['Solution 1: ...', 'Solution 2: ...']"
            )
        
        if len(solutions) < 2:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                f"Error: You must propose at least 2 solutions, but provided {len(solutions)}.\n"
                "Requirements:\n"
                "  • Each solution must be meaningfully different (different approach/algorithm)\n"
                "  • Each solution must be accurate and solve the problem\n"
                "  • Explain why each solution is valid\n"
                "  • Then select the best one with reasoning\n\n"
                "Example:\n"
                "solutions=[\n"
                "  'Solution 1: Use recursion approach because...',\n"
                "  'Solution 2: Use iterative approach because...'\n"
                "]"
            )

        self.is_solution_approved = True
        return "Approved"
          
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            file_path, *name_parts = function_path.split("::")
            function_name = "::".join(name_parts)
            try:
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
                visitor = FunctionVisitor(content)
                visitor.visit(ast.parse(content, filename=file_path))
                functions[function_path] = visitor.functions.get(function_name, {}).get("body", f"Function {function_name} not found in {file_path}")
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {e}"
        return functions

    @EnhancedToolManager.tool
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            file_path, *name_parts = class_path.split("::")
            class_name = "::".join(name_parts)
            try:
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
                visitor = ClassVisitor(content)
                visitor.visit(ast.parse(content, filename=file_path))
                classes[class_path] = visitor.classes.get(class_name, {}).get("body", f"Class {class_name} not found in {file_path}")
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {e}"
        return classes

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project.
        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE
        for root, _, files in os.walk("."):
            if ".git" in root: continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for i, line in enumerate(f, 1):
                                if re.search(search_term, line, search_flags):
                                    output.append(f"{file_path}:{i}:{line.strip()}")
                    except Exception:
                        continue
        limited_output = Utils.limit_strings("\n".join(output), n=100)
        if not limited_output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND, f"'{search_term}' not found in the codebase.")
        return limited_output

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: source = f.read()
            tree = ast.parse(source, filename=file_path)
        except Exception as e:
            logger.error(f"Could not parse {file_path}: {e}")
            return []
        
        func_ranges: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, 'lineno', None)
                end = getattr(node, 'end_lineno', None)
                if start and end: func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: source_lines = f.read().splitlines()
        match_lines_indices = [i for i, line in enumerate(source_lines) if search_term in line]
        if not match_lines_indices:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND, f"'{search_term}' not found in '{file_path}'")

        func_ranges = self.get_function_ranges(file_path)
        def _containing_function(line_idx: int):
            for start, end, name in func_ranges:
                if start - 1 <= line_idx < end: return (start, end, name)
            return None

        functions_to_return = {info for ln_idx in match_lines_indices if (info := _containing_function(ln_idx))}
        standalone_lines = [i for i in match_lines_indices if not _containing_function(i)]
        
        chunks = [f"(lines {start}-{end}):\n" + "\n".join(source_lines[start - 1:end]) for start, end, name in sorted(list(functions_to_return))]
        chunks.extend(f"{i+1}:{source_lines[i]}" for i in standalone_lines)
        
        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH, f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)
    
    @EnhancedToolManager.tool
    def start_over(self,problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("="*20 + "STARTING OVER" + "="*20)
        os.system("git reset --hard")
        logger.info(f"Problem with old approach: {problem_with_old_approach}")
        logger.info(f"New approach to try: {new_apprach_to_try}")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH,f"Error: file '{file_path}' is not a python file.")
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name, exist_ok=True)
        test_fn = (test_function_code or "").strip()
        if not test_fn: raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,"Error: test_function_code cannot be empty.")
        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: generated test function has syntax error: {err}")
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: inserting test caused syntax error. First error: {first_error}")


        self._save(file_path, new_content) # Assuming new_content is derived correctly
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files: self.generated_test_files.append(rel)
        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Now includes intelligent test failure analysis to help debug issues faster.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files with analysis if tests fail.
        '''
        if not file_paths: return "No file paths provided to test."

        if self.test_runner == "pytest":
            cmd = ["pytest"] + file_paths
        elif self.test_runner == "unittest":
            # unittest is a Python module, not a standalone command
            if self.test_runner_mode == "MODULE":
                args = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
            else:
                args = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
            cmd = ["python", "-m", "unittest"] + args
        else:
            # Other test runners
            if self.test_runner_mode == "MODULE":
                args = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
            else:
                args = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
            cmd = [self.test_runner] + args

        logger.info(f"Running test command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        test_output = (result.stdout or "") + (result.stderr or "")

        # If tests failed, add intelligent analysis
        if result.returncode != 0 and ("FAILED" in test_output or "ERROR" in test_output or "AssertionError" in test_output):
            try:
                logger.info("[TEST] Tests failed, performing intelligent analysis...")
                analysis = analyze_test_failure(test_output)

                # Check for unit conversion issues and add prominent warning
                unit_warning = ""
                numeric_analysis = analysis.get('numeric_analysis', {})
                if numeric_analysis and numeric_analysis.get('likely_unit_issue') == 'yes':
                    unit_warning = f"""
⚠️⚠️⚠️ UNIT CONVERSION ISSUE DETECTED! ⚠️⚠️⚠️
{numeric_analysis.get('exact_fix', 'Check if you need to multiply or divide by a conversion factor')}
This is a COMMON error - check if your function returns the right units!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

                analysis_summary = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 INTELLIGENT TEST FAILURE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{unit_warning}
📊 Failure Category: {analysis.get('failure_category', 'UNKNOWN')}
🎯 Confidence: {analysis.get('confidence', 'unknown')}

💡 Root Cause:
{analysis.get('root_cause', 'Unknown')}

📝 Expected vs Actual:
{analysis.get('expected_vs_actual', 'N/A')}

🔧 Fix Guidance:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(analysis.get('fix_guidance', [])))}

📍 Likely Location: {analysis.get('likely_location', 'unknown')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
                return test_output + analysis_summary

            except Exception as e:
                logger.error(f"[TEST] Failed to analyze test failure: {e}")
                return test_output

        return test_output

    @EnhancedToolManager.tool
    def run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in.
        Output:
            Returns the stdout/stderr from the executed file.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        return self._run_code(content, file_path)
    
    def _validate_python_syntax(self, code: str, file_path: str) -> tuple[bool, str]:
        """
        Validates Python code syntax before saving.

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}\n{e.text}"
            logger.error(f"[VALIDATION] Syntax error in {file_path}: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Code validation error: {str(e)}"
            logger.error(f"[VALIDATION] Error validating {file_path}: {error_msg}")
            return False, error_msg

    @EnhancedToolManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files with automatic syntax validation.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL, "Error: You must get approval before applying edits. Call get_approval_for_solution tool first.")
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND, f"Error: file '{file_path}' does not exist.")

        with open(file_path, 'r', encoding='utf-8') as f: original = f.read()
        
        if original.count(search) != 1:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND if original.count(search) == 0 else EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND,
                f"Error: search string found {original.count(search)} times in '{file_path}'. Must be found exactly once.")
        
        new_content = original.replace(search, replace, 1)
        self._save(file_path, new_content) # _save includes syntax check
        return "ok, code edit applied successfully"

    @EnhancedToolManager.tool
    def compare_test_results_before_after(self, test_files: List[str]) -> str:
        '''
        Compare test results before and after your changes to detect regressions.
        This runs tests on the initial checkpoint and current state, then compares results.
        USE THIS BEFORE calling finish() to ensure no passing tests were broken!

        Arguments:
            test_files: list of test file paths to run

        Output:
            Detailed comparison showing:
            - New failures (REGRESSIONS - tests that passed before but fail now)
            - New passes (tests that failed before but now pass)
            - Still failing tests
            - Still passing tests
        '''
        logger.info(f"Comparing test results for files: {test_files}")

        if not hasattr(self, 'initial_checkpoint_created') or not self.initial_checkpoint_created:
            return "Error: No initial checkpoint found. Cannot compare test results."

        # Run tests on current state
        logger.info("Running tests on current state...")
        current_output = self._run_tests(test_files)

        # Switch to initial checkpoint
        logger.info("Switching to initial checkpoint...")
        switch_result = GitCheckpointManager.switch_checkpoint(".", "agent_initial_state", save_current=True)
        if switch_result['status'] != 'success':
            return f"Error switching to initial checkpoint: {switch_result['message']}"

        # Run tests on initial state
        logger.info("Running tests on initial state...")
        initial_output = self._run_tests(test_files)

        # Restore current state
        logger.info("Restoring current state...")
        GitCheckpointManager.restore_stashed_changes(".", 0, remove_after_apply=True)

        # Compare results
        comparison = EnhancedTestComparator.compare_test_results(initial_output, current_output)

        # Format output
        result = "📊 TEST COMPARISON RESULTS:\n\n"

        if comparison['new_failures']:
            result += f"❌ NEW FAILURES (REGRESSIONS - {len(comparison['new_failures'])}):\n"
            result += "These tests PASSED before but FAIL now - you MUST fix these!\n"
            for test in comparison['new_failures']:
                result += f"  - {test}\n"
            result += "\n"

            # Extract failure details
            failures = EnhancedTestComparator.extract_failure_details(current_output)
            if failures:
                result += "Failure Details:\n"
                for failure in failures[:3]:  # Show first 3 failures
                    result += f"{failure}\n\n"

        if comparison['new_passes']:
            result += f"✅ NEW PASSES ({len(comparison['new_passes'])}):\n"
            result += "These tests FAILED before but PASS now - good work!\n"
            for test in comparison['new_passes']:
                result += f"  - {test}\n"
            result += "\n"

        if comparison['still_failing']:
            result += f"⚠️ STILL FAILING ({len(comparison['still_failing'])}):\n"
            for test in comparison['still_failing'][:5]:  # Show first 5
                result += f"  - {test}\n"
            if len(comparison['still_failing']) > 5:
                result += f"  ... and {len(comparison['still_failing']) - 5} more\n"
            result += "\n"

        if comparison['still_passing']:
            result += f"✓ STILL PASSING ({len(comparison['still_passing'])}): Good!\n\n"

        # Final verdict
        if comparison['new_failures']:
            result += "⛔ VERDICT: CANNOT FINISH - You introduced regressions! Fix the new failures first.\n"
        elif comparison['new_passes'] or not comparison['still_failing']:
            result += "✅ VERDICT: READY TO FINISH - No regressions detected!\n"
        else:
            result += "⚠️ VERDICT: Tests still failing, but no new regressions. Review if this is acceptable.\n"

        return result

    @EnhancedToolManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: A detailed summary of the problem, investigation, and solution.
        '''
        logger.info(f"Finish called. Summary: {investigation_summary}")
        return "finish"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _guess_tokens(text: str) -> int:
    """Rough token count estimate: ~0.75 tokens per word."""
    return int(len(text.split()) * 0.75)

# ==========================================================================================
# CONTEXT WINDOW MANAGEMENT - Prevent token limit overflow
# ==========================================================================================

def calculate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Calculate total tokens in message list.
    Uses improved token estimation accounting for JSON structure and special tokens.
    """
    total_tokens = 0
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Role tokens (~5 tokens per message for structure)
        total_tokens += 5
        
        # Content tokens (improved estimation)
        if isinstance(content, str):
            # Account for special tokens, punctuation, code syntax
            words = content.split()
            # Code has more tokens per word due to symbols
            if any(keyword in content for keyword in ['def ', 'class ', 'import ', '{', '}', '(', ')']):
                total_tokens += int(len(words) * 1.0)  # Code: ~1 token per word
            else:
                total_tokens += int(len(words) * 0.75)  # Text: ~0.75 tokens per word
    
    return total_tokens

def get_context_limit_for_model(model: str) -> int:
    """Get the context window limit for a specific model."""
    return AgentConfig.MODEL_CONTEXT_LIMITS.get(model, AgentConfig.DEFAULT_CONTEXT_LIMIT)

def get_available_tokens_for_input(model: str) -> int:
    """
    Calculate how many tokens are available for input after reserving space for output.
    Applies safety margin to prevent edge-case failures.
    """
    total_limit = get_context_limit_for_model(model)
    # Reserve tokens for output
    available = total_limit - AgentConfig.MAX_OUTPUT_TOKENS
    # Apply safety margin
    safe_available = int(available * AgentConfig.CONTEXT_SAFETY_MARGIN)
    return safe_available

def truncate_messages_to_fit(messages: List[Dict[str, Any]], model: str, 
                             preserve_system: bool = True) -> List[Dict[str, Any]]:
    """
    Intelligently truncate messages to fit within context window.
    
    Strategy:
    1. Always preserve system prompt (if preserve_system=True)
    2. Always preserve most recent user message
    3. Truncate from the middle of conversation history
    4. Preserve structure and coherence
    """
    max_tokens = get_available_tokens_for_input(model)
    current_tokens = calculate_message_tokens(messages)
    
    if current_tokens <= max_tokens:
        # No truncation needed
        print(f"[CONTEXT] Messages fit: {current_tokens}/{max_tokens} tokens")
        return messages
    
    print(f"[CONTEXT] WARNING: Messages exceed limit: {current_tokens}/{max_tokens} tokens")
    print(f"[CONTEXT] Truncating {current_tokens - max_tokens} tokens...")
    
    # Separate messages by type
    system_msgs = [msg for msg in messages if msg.get('role') == 'system']
    user_msgs = [msg for msg in messages if msg.get('role') == 'user']
    assistant_msgs = [msg for msg in messages if msg.get('role') == 'assistant']
    
    # Start with essential messages
    truncated = []
    tokens_used = 0
    
    # 1. Add system prompt (always preserve if requested)
    if preserve_system and system_msgs:
        system_msg = system_msgs[0]  # Use first system message
        system_tokens = calculate_message_tokens([system_msg])
        truncated.append(system_msg)
        tokens_used += system_tokens
    
    # 2. Reserve space for most recent user message
    if user_msgs:
        last_user_msg = user_msgs[-1]
        last_user_tokens = calculate_message_tokens([last_user_msg])
        tokens_reserved = tokens_used + last_user_tokens
    else:
        last_user_msg = None
        tokens_reserved = tokens_used
    
    # 3. Calculate remaining budget
    remaining_budget = max_tokens - tokens_reserved
    
    # 4. Add middle messages (conversation history) from most recent backwards
    middle_messages = []
    for msg in reversed(messages):
        # Skip system (already added) and last user (will add at end)
        if msg.get('role') == 'system' or (msg == last_user_msg):
            continue
        
        msg_tokens = calculate_message_tokens([msg])
        if tokens_used + msg_tokens <= remaining_budget:
            middle_messages.insert(0, msg)  # Insert at beginning to maintain order
            tokens_used += msg_tokens
        else:
            # Budget exhausted
            break
    
    truncated.extend(middle_messages)
    
    # 5. Add most recent user message
    if last_user_msg:
        truncated.append(last_user_msg)
        tokens_used += last_user_tokens
    
    print(f"[CONTEXT] Truncated to {len(truncated)}/{len(messages)} messages ({tokens_used}/{max_tokens} tokens)")
    
    return truncated

def smart_truncate_content(content: str, max_tokens: int) -> str:
    """
    Intelligently truncate content to fit token limit.
    Preserves beginning and end, truncates middle.
    """
    current_tokens = _guess_tokens(content)
    if current_tokens <= max_tokens:
        return content
    
    # Calculate how much to keep
    keep_ratio = max_tokens / current_tokens
    
    # Keep first 60% and last 40% of available space
    chars_to_keep = int(len(content) * keep_ratio)
    first_part_size = int(chars_to_keep * 0.6)
    last_part_size = int(chars_to_keep * 0.4)
    
    first_part = content[:first_part_size]
    last_part = content[-last_part_size:]
    
    truncation_notice = f"\n\n... [TRUNCATED {current_tokens - max_tokens} tokens] ...\n\n"
    
    return first_part + truncation_notice + last_part

ZERO_VEC: List[float] = [0.0] * 1024

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)



# ============================================================================
# MAIN AGENT LOGIC AND ENTRYPOINT
# ============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Main entry point for the agent."""
    global RUN_ID, run_id, REPO_DIR
    RUN_ID = os.getenv("RUN_ID", "")
    run_id = RUN_ID
    repo_dir = os.path.abspath(repo_dir)
    REPO_DIR = repo_dir
    
    timeout = 1000 if test_mode else AgentConfig.DEFAULT_TIMEOUT
    max_steps = AgentConfig.MAX_TEST_PATCH_TIMEOUT if test_mode else AgentConfig.MAX_FIX_TASK_STEPS

    print("\n" + "="*80)
    print("🚀 AGENT STARTING")
    print("="*80)
    logger.info(f"[AGENT-START] Run ID: {RUN_ID}")
    logger.info(f"[AGENT-START] Repo directory: {repo_dir}")
    logger.info(f"[AGENT-START] Test mode: {test_mode}")
    logger.info(f"[AGENT-START] Timeout: {timeout}s, Max steps: {max_steps}")

    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir): os.chdir(repo_dir)
    ensure_git_initialized()
    set_env_for_agent()

    problem_statement = input_dict.get("problem_statement")
    logger.info(f"[AGENT-START] Problem statement length: {len(problem_statement)} chars")
    
    print("\n" + "─"*80)
    print("🔍 DETERMINING PROBLEM TYPE...")
    print("─"*80)
    problem_type = check_problem_type(problem_statement)
    logger.info(f"[PROBLEM-TYPE] Determined: {problem_type}")
    print(f"✅ Problem type: {problem_type}")
    
    if problem_type == AgentConfig.PROBLEM_TYPE_FIX:
        print("\n" + "─"*80)
        print("🎯 FIX TASK DETECTED - STARTING ITERATIVE WORKFLOW")
        print("─"*80)
        logger.info("[WORKFLOW] Starting iterative FIX workflow")
        result = process_fix_task(input_dict)
        print("\n" + "─"*80)
        print("✅ ITERATIVE WORKFLOW COMPLETED")
        print("─"*80)
        logger.info("[WORKFLOW] Iterative workflow completed")
    else:
        print("\n" + "─"*80)
        print("🆕 CREATE TASK DETECTED - STARTING CREATE WORKFLOW")
        print("─"*80)
        logger.info("[WORKFLOW] Starting CREATE task workflow")
        result = process_create_task(input_dict)
        print("\n" + "─"*80)
        print("✅ CREATE WORKFLOW COMPLETED")
        print("─"*80)
        logger.info("[WORKFLOW] CREATE workflow completed")

    os.system("git reset --hard")
    print("\n" + "="*80)
    print("🏁 AGENT FINISHED")
    print("="*80)
    logger.info("[AGENT-END] Agent execution completed")
    return result





def analyze_problem_requirements(problem_statement: str) -> dict:
    """
    Performs deep analysis of problem requirements to extract critical information.
    This helps the agent understand EXACTLY what is expected.

    Args:
        problem_statement: The problem description

    Returns:
        Dictionary with extracted requirements
    """
    logger.info("[ANALYSIS] Starting deep problem analysis")

    analysis_prompt = textwrap.dedent("""
    You are an expert problem analyst. Analyze the problem statement and extract ALL critical requirements.
    Your analysis will guide the solution implementation, so be PRECISE and THOROUGH.

    Extract the following in a structured format:

    1. **Expected Output Type** (CRITICAL):
       - Analyze ALL examples to determine exact type (int, float, str, list, dict, etc.)
       - Check decimal points in examples (8.00 → float, 800 → int)
       - Identify units if any (dollars vs cents, meters vs cm, etc.)

    2. **Input Constraints**:
       - Types and ranges of inputs
       - Edge cases mentioned (empty, None, zero, negative, etc.)

    3. **Exact Error Messages** (if any):
       - Extract EXACT error messages from problem statement
       - Note which exceptions should be raised and when

    4. **Core Requirements**:
       - What calculations/logic are needed?
       - What algorithms might work?
       - What are the success criteria?

    5. **Edge Cases to Handle**:
       - Empty inputs
       - Boundary values
       - Special cases mentioned

    6. **Example Analysis**:
       - List all input→output examples
       - What patterns do they show?
       - What type/format is consistent across all?

    Output your analysis as a JSON object with these keys:
    {
      "output_type": "int|float|str|list|dict|etc",
      "output_format": "description of format",
      "units": "description of units if applicable",
      "error_messages": ["exact error message 1", "exact error message 2"],
      "core_logic": "description of required logic",
      "edge_cases": ["case 1", "case 2"],
      "examples": [{"input": "...", "output": "...", "type": "..."}],
      "critical_notes": ["note 1", "note 2"]
    }

    Only output the JSON, nothing else.
    """)

    messages = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": f"Problem statement to analyze:\n\n{problem_statement}"}
    ]

    try:
        response = EnhancedNetwork.make_request(
            messages,
            model=AgentConfig.DEEPSEEK_MODEL_NAME,  # Use DeepSeek for analysis
            temperature=0.0
        )

        # Extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        requirements = json.loads(response)
        logger.info(f"[ANALYSIS] Extracted requirements: {json.dumps(requirements, indent=2)}")
        return requirements

    except Exception as e:
        logger.error(f"[ANALYSIS] Failed to parse requirements: {e}")
        # Return empty dict if analysis fails
        return {
            "output_type": "unknown",
            "output_format": "unknown",
            "units": "",
            "error_messages": [],
            "core_logic": "",
            "edge_cases": [],
            "examples": [],
            "critical_notes": []
        }


def analyze_test_failure(test_output: str, problem_statement: str = "") -> dict:
    """
    Analyzes test failure output to categorize the issue and provide actionable guidance.

    Args:
        test_output: The test failure output
        problem_statement: Optional problem statement for context

    Returns:
        Dictionary with failure analysis
    """
    logger.info("[TEST_ANALYSIS] Analyzing test failures")

    analysis_prompt = textwrap.dedent("""
    You are an expert test failure analyst. Analyze the test output and categorize the failures.

    **Analyze these aspects:**

    1. **Failure Category** (Choose primary one):
       - TYPE_MISMATCH: Wrong return type (int vs float, str vs list, etc.)
       - VALUE_ERROR: Wrong calculated value but correct type
       - ERROR_MESSAGE_MISMATCH: Wrong exception message text
       - MISSING_ERROR: Should raise exception but doesn't
       - WRONG_EXCEPTION: Raises wrong exception type
       - LOGIC_ERROR: Algorithmic/logic bug
       - EDGE_CASE_FAILURE: Fails on edge cases (empty, None, boundary)
       - IMPORT_ERROR: Missing imports or module issues
       - SYNTAX_ERROR: Code syntax problems
       - OTHER: Other issue

    2. **Root Cause Analysis**:
       - What EXACTLY is wrong?
       - Which line/function is the issue?
       - What was expected vs actual?

    3. **Type Analysis** (if TYPE_MISMATCH):
       - Expected type: ?
       - Actual type: ?
       - Unit issue: (e.g., dollars vs cents, meters vs cm)?
       - **CRITICAL CHECK**: Is actual exactly 100x smaller than expected? (e.g., 145.6 vs 14560 → cents vs dollars!)
       - **CRITICAL CHECK**: Is actual exactly 100x larger than expected? (e.g., 14560 vs 145.6 → multiplied when should divide!)

    4. **Numeric Mismatch Analysis**:
       - If expected and actual are both numbers:
         • Ratio: actual / expected = ?
         • Is ratio ≈ 0.01 (÷100 issue)?
         • Is ratio ≈ 100 (×100 issue)?
         • Is ratio ≈ 1000 (unit conversion issue)?
         • Extract the EXACT conversion needed

    5. **Actionable Fix Guidance**:
       - Where to look (specific files/functions)?
       - What to change?
       - What to check?
       - If unit conversion detected: Provide EXACT fix (multiply by X or divide by Y)

    Output as JSON:
    {
      "failure_category": "TYPE_MISMATCH|VALUE_ERROR|etc",
      "root_cause": "detailed explanation",
      "expected_vs_actual": "what was expected vs what was returned",
      "type_issue": {"expected": "type", "actual": "type", "unit_conversion": "yes/no", "conversion_ratio": "number"},
      "numeric_analysis": {"ratio": "number", "likely_unit_issue": "yes/no", "exact_fix": "multiply by X or divide by Y"},
      "fix_guidance": ["step 1", "step 2", "step 3"],
      "likely_location": "description of where bug likely is",
      "confidence": "high|medium|low"
    }

    Only output JSON, nothing else.
    """)

    messages = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": f"Test output to analyze:\n\n{test_output}\n\nProblem statement (for context):\n{problem_statement[:1000] if problem_statement else 'N/A'}"}
    ]

    try:
        response = EnhancedNetwork.make_request(
            messages,
            model=AgentConfig.DEEPSEEK_MODEL_NAME,
            temperature=0.0
        )

        # Extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        analysis = json.loads(response)
        logger.info(f"[TEST_ANALYSIS] Category: {analysis.get('failure_category')}, Confidence: {analysis.get('confidence')}")
        return analysis

    except Exception as e:
        logger.error(f"[TEST_ANALYSIS] Failed to analyze: {e}")
        return {
            "failure_category": "UNKNOWN",
            "root_cause": "Could not analyze test failure",
            "expected_vs_actual": "",
            "type_issue": {},
            "fix_guidance": ["Review test output manually", "Check for common errors"],
            "likely_location": "unknown",
            "confidence": "low"
        }


def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": AgentConfig.PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"}
            ]

            response = EnhancedNetwork.make_request(messages, model=AgentConfig.QWEN_MODEL_NAME)

            if response not in [AgentConfig.PROBLEM_TYPE_CREATE, AgentConfig.PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1

        time.sleep(2)

    return response




def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:
        ```text
        This is a test.
        This is another test!
        ```text
        Then the text block should be:
        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    # Pattern to match ```text...``` blocks
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        
        return f'```text\n{processed_content}\n```'
    
    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str, requirements: dict = None) -> str:
    retry = 0

    # Add requirements analysis if available
    requirements_text = ""
    if requirements and requirements.get("output_type") != "unknown":
        requirements_text = f"""

**🔍 CRITICAL REQUIREMENTS EXTRACTED (Follow These EXACTLY):**

- **Output Type**: {requirements.get('output_type', 'unknown')}
- **Output Format**: {requirements.get('output_format', 'unknown')}
- **Units**: {requirements.get('units', 'N/A')}
- **Error Messages**: {json.dumps(requirements.get('error_messages', []))}
  ⚠️ Use these EXACT error messages in your code
- **Core Logic**: {requirements.get('core_logic', '')}
- **Edge Cases to Handle**: {json.dumps(requirements.get('edge_cases', []))}
- **Examples**: {json.dumps(requirements.get('examples', []))}
- **Critical Notes**: {json.dumps(requirements.get('critical_notes', []))}

⚠️ **MOST COMMON FAILURE**: Wrong output type!
- If examples show decimals (8.00), return FLOAT not INT
- If examples show integers (800), return INT not FLOAT
- If problem mentions "dollars" check if examples use dollars (float) or cents (int)
- Match the EXACT type from examples
"""

    code_generation_messages = [
        {
            "role": "system",
            "content": AgentConfig.GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n{requirements_text}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(
                code_generation_messages, 
                model=AgentConfig.QWEN_MODEL_NAME,
                temperature=AgentConfig.REASONING_EFFORT["high"]  # High effort for complex solution generation
            )

            solution = code_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    retry = 0

    # First, analyze problem requirements
    logger.info("Step 1: Analyzing problem requirements")
    requirements = analyze_problem_requirements(problem_statement)

    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation with requirements")

            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton, requirements)

            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": AgentConfig.GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(
                    messages, 
                    model=AgentConfig.QWEN_MODEL_NAME,
                    temperature=AgentConfig.REASONING_EFFORT["high"]  # High effort for solution generation
                )
                
                # Clean up the response
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                logger.info("Generated initial solution successfully using fallback approach")
                return solution
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str, requirements: dict = None) -> str:
    retry = 0

    # Add requirements analysis if available
    requirements_text = ""
    if requirements and requirements.get("output_type") != "unknown":
        requirements_text = f"""

**🔍 CRITICAL TEST REQUIREMENTS (Match These EXACTLY):**

- **Expected Output Type**: {requirements.get('output_type', 'unknown')}
  ⚠️ Your assertions MUST expect this exact type!
- **Output Format**: {requirements.get('output_format', 'unknown')}
- **Units**: {requirements.get('units', 'N/A')}
- **Expected Error Messages**: {json.dumps(requirements.get('error_messages', []))}
  ⚠️ Use assertRaises and check for these EXACT error messages
- **Examples to Test**: {json.dumps(requirements.get('examples', []))}
  ⚠️ Include ALL these examples as test cases
- **Edge Cases to Cover**: {json.dumps(requirements.get('edge_cases', []))}

⚠️ **CRITICAL**: Match the output type from examples!
- If problem shows `8.00`, write `assertEqual(func(), 8.00)` NOT `assertEqual(func(), 800)`
- If problem shows `800`, write `assertEqual(func(), 800)` NOT `assertEqual(func(), 8.00)`
"""

    test_generation_messages = [
        {
            "role": "system",
            "content": AgentConfig.GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n{requirements_text}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=AgentConfig.QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            
            # Step 5: Infinite Loop Check and Validation
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": AgentConfig.TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=AgentConfig.QWEN_MODEL_NAME)
            logger.info("Step 2 - Testcase check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    
    return ""

def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str, requirements: dict = None) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")

            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton, requirements)
            
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": AgentConfig.GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=AgentConfig.QWEN_MODEL_NAME)
                
                # Clean up the response
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def validate_solution_quality(code: str, problem_statement: str = "") -> dict:
    """
    Validates the quality of generated code before committing it.

    Returns:
        Dictionary with validation results
    """
    logger.info("[VALIDATION] Checking solution quality")

    validation_prompt = textwrap.dedent("""
    You are a code quality validator. Analyze the generated code for common issues.

    **Check for:**

    1. **Type Consistency**: Are return types consistent? Any int/float confusion?
    2. **Error Handling**: Are exceptions handled? Are error messages present if needed?
    3. **Edge Cases**: Does code handle empty inputs, None, boundary values?
    4. **Logic Errors**: Any obvious algorithmic bugs?
    5. **Completeness**: Are all required functions implemented?
    6. **Code Quality**: Any obvious anti-patterns?

    **Rate each aspect:**
    - PASS: Looks good
    - WARNING: Minor issues
    - FAIL: Critical issues

    Output as JSON:
    {
      "type_consistency": {"status": "PASS|WARNING|FAIL", "issues": ["issue1", ...]},
      "error_handling": {"status": "PASS|WARNING|FAIL", "issues": []},
      "edge_cases": {"status": "PASS|WARNING|FAIL", "issues": []},
      "logic": {"status": "PASS|WARNING|FAIL", "issues": []},
      "completeness": {"status": "PASS|WARNING|FAIL", "issues": []},
      "overall_score": "PASS|WARNING|FAIL",
      "recommendations": ["rec1", "rec2"]
    }

    Only output JSON.
    """)

    messages = [
        {"role": "system", "content": validation_prompt},
        {"role": "user", "content": f"Code to validate:\n\n{code}\n\nProblem statement:\n{problem_statement[:500]}"}
    ]

    try:
        response = EnhancedNetwork.make_request(
            messages,
            model=AgentConfig.DEEPSEEK_MODEL_NAME,
            temperature=0.0
        )

        # Extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        validation = json.loads(response)
        logger.info(f"[VALIDATION] Overall score: {validation.get('overall_score', 'UNKNOWN')}")
        return validation

    except Exception as e:
        logger.error(f"[VALIDATION] Failed to validate: {e}")
        return {"overall_score": "WARNING", "recommendations": ["Manual review needed"]}


def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files




    
def process_create_task(input_dict):
    print("\n" + "┌"+ "─"*78 + "┐")
    print("│ 🆕 CREATE TASK WORKFLOW STARTING" + " "*44 + "│")
    print("└"+ "─"*78 + "┘")
    logger.info("[CREATE] Starting CREATE task workflow")

    start_time = time.time()
    problem_statement = post_process_instruction(input_dict.get("problem_statement", ""))
    logger.info(f"[CREATE] Problem statement length: {len(problem_statement)} chars")

    logger.info("[CREATE] Step 0: Analyzing problem requirements")
    requirements = analyze_problem_requirements(problem_statement)
    logger.info(f"[CREATE] Requirements analysis complete")

    logger.info("[CREATE] Step 1: Generating code skeleton")
    code_skeleton = get_code_skeleton()
    logger.info(f"[CREATE] Code skeleton generated: {len(code_skeleton)} chars")

    logger.info("[CREATE] Step 2: Generating initial solution")
    initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    logger.info(f"[CREATE] Initial solution generated: {len(initial_solution)} chars")

    logger.info("[CREATE] Step 3: Extracting and writing solution files")
    created_files = extract_and_write_files(initial_solution)
    logger.info(f"[CREATE] Created {len(created_files)} solution files")
    print(f"✅ Created {len(created_files)} solution files")

    logger.info("[CREATE] Step 4: Generating test files with requirements")
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton, requirements)
    logger.info(f"[CREATE] Test cases generated: {len(test_cases)} chars")
    
    logger.info("[CREATE] Step 5: Extracting and writing test files")
    test_files = extract_and_write_files(test_cases)
    logger.info(f"[CREATE] Created {len(test_files)} test files")
    print(f"✅ Created {len(test_files)} test files")

    time_spent = time.time() - start_time
    remaining_timeout = AgentConfig.DEFAULT_TIMEOUT - time_spent - 60
    logger.info(f"[CREATE] Time spent: {time_spent:.1f}s, Remaining timeout: {remaining_timeout:.1f}s")
    
    logger.info("[CREATE] Step 6: Starting iterative FIX workflow for refinement")
    print("\n" + "─"*80)
    print("🔧 Starting iterative refinement workflow...")
    print("─"*80)
    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=remaining_timeout,
        run_id_1=run_id,
        instance_id="",
        test_runner="unittest",
        test_runner_mode="FILE",
        n_max_steps=50 
    )

    if patch is None:
        logger.warning("Fix task workflow did not produce a patch. Using initial solution.")
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    return tool_manager.get_final_git_patch()



def get_code_skeleton() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []
    
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            
            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            
            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                
                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                
                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                
                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
            except PermissionError:
                # Handle directories we can't read
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)
    
    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None

def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": AgentConfig.FIND_TEST_RUNNER_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=AgentConfig.DEEPSEEK_MODEL_NAME)
        return response.strip() or "pytest"
    except Exception as e:
        logger.error(f"Error finding test runner: {e}")
        return "pytest"

def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')

def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path

def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": AgentConfig.TEST_RUNNER_MODE_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=AgentConfig.DEEPSEEK_MODEL_NAME)
        return response.strip() or "FILE"
    except Exception as e:
        logger.error(f"Error determining test runner mode: {e}")
        return "FILE"

def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)
    
    except (FileNotFoundError, UnicodeDecodeError):
        return 0

def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        print(f"no test file found")
        return "pytest", "FILE"

    print(f"test_file_path: {test_file_path}")
    readme_file_path = find_readme(test_file_path, '.')
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)
    else:
        print("No README found, using default pytest")

    return test_runner, test_runner_mode



def process_fix_task(input_dict: Dict[str, Any]):
    global RUN_ID, REPO_DIR
    #RUN_ID = os.getenv("RUN_ID", "")
    # Set up environment
    RUN_ID = os.getenv("RUN_ID", "")
    repo_dir = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_dir.split('/')[-1]
    repod_path = repo_dir[:-len(repod_dir)-1]
    
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    
    REPO_DIR = os.getcwd()

    set_env_for_agent()
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        logger.info("Attempting iterative FIX workflow...")
        result = fix_task_solve_workflow(
            input_dict.get("problem_statement", ""),
            timeout=AgentConfig.DEFAULT_TIMEOUT,
            run_id_1=RUN_ID,
            instance_id="",
            test_runner="pytest",
            test_runner_mode="FILE"
        )
        return result
    except Exception as e:
        logger.critical(f"Iterative FIX workflow failed critically: {e} {traceback.format_exc()}")
        return ""

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps=AgentConfig.MAX_FIX_TASK_STEPS) -> str:
    global run_id
    run_id = run_id_1

    print("\n" + "┌"+ "─"*78 + "┐")
    print("│ 🔧 ITERATIVE FIX WORKFLOW STARTING" + " "*42 + "│")
    print("└"+ "─"*78 + "┘")
    logger.info("[ITERATIVE] Starting iterative FIX workflow")
    logger.info(f"[ITERATIVE] Run ID: {run_id}")
    logger.info(f"[ITERATIVE] Max steps: {n_max_steps}, Timeout: {timeout}s")
    logger.info(f"[ITERATIVE] Test runner: {test_runner} (mode: {test_runner_mode})")

    cot = EnhancedCOT.load_state(AgentConfig.STATE_FILE_PATH)
    logger.info(f"[ITERATIVE] Loaded COT state: {len(cot.thoughts)} existing thoughts")
    
    available_tools_list = [
            "get_file_content", "save_file", "get_approval_for_solution",
            "get_functions", "get_classes", "search_in_all_files_content",
            "search_in_specified_file_v2", "start_over", "run_repo_tests",
            "run_code", "apply_code_edit", "generate_test_function",
            "compare_test_results_before_after", "finish"
    ]
    
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=available_tools_list,
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"[ITERATIVE] Initialized tool manager with {len(available_tools_list)} tools")

    # Create initial checkpoint for before/after test comparison
    print("📸 Creating initial checkpoint...")
    checkpoint_result = GitCheckpointManager.create_checkpoint(".", "agent_initial_state")
    if checkpoint_result['status'] == 'success':
        tool_manager.initial_checkpoint_created = True
        logger.info(f"[CHECKPOINT] Created initial checkpoint: {checkpoint_result['message']}")
        print(f"✅ Initial checkpoint created: {checkpoint_result['commit_hash'][:8]}")
    else:
        tool_manager.initial_checkpoint_created = False
        logger.warning(f"[CHECKPOINT] Failed to create initial checkpoint: {checkpoint_result['message']}")
        print(f"⚠️ Checkpoint creation failed (will continue without before/after comparison)")

    system_prompt = AgentConfig.FIX_TASK_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(), format_prompt=AgentConfig.FORMAT_PROMPT_V0)
    instance_prompt = AgentConfig.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logger.info(f"[ITERATIVE] Workflow started at {time.strftime('%H:%M:%S')}")
    print(f"🚀 Starting iterative workflow (max {n_max_steps} steps, {timeout}s timeout)")

    failure_tracker = {"tool_name": None, "args_hash": None, "count": 0}
    is_stuck = False
    
    # Enhanced loop detection - track recent actions (both successful and failed)
    recent_actions = []
    MAX_IDENTICAL_ACTIONS = 2
    consecutive_identical_count = 0
    last_action = None

    start_step = len(cot.thoughts)
    for step in range(start_step, n_max_steps):
        logger.info(f"[RUN:{run_id}] Step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            logger.warning(f"Global timeout of {timeout}s reached. Aborting.")
            cot.add_action(EnhancedCOT.Action("global timeout reached", "", {}, "", is_error=True))
            break

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())

        if is_stuck:
            messages.append({"role": "user", "content": AgentConfig.STUCK_DETECTION_PROMPT})
            is_stuck = False

        if cot.is_thought_repeated():
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": AgentConfig.DO_NOT_REPEAT_TOOL_CALLS.format(
                previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
            )})
    
        messages.append({"role": "system", "content": AgentConfig.STOP_INSTRUCTION})

        # Apply context window management before inference
        messages = truncate_messages_to_fit(messages, AgentConfig.GLM_MODEL_NAME, preserve_system=True)

        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(messages, model=AgentConfig.GLM_MODEL_NAME, run_id=run_id)
            
            # FIX 3: Garbage Detection - Validate LLM response
            if not raw_text or len(raw_text.strip()) < 10:
                logger.warning("Empty or invalid LLM response detected - skipping this step")
                continue
                
        except Exception as e:
            error_msg = f"Inference Error: {e} {traceback.format_exc()}"
            logger.error(error_msg)
            cot.add_action(EnhancedCOT.Action(error_msg, "", {}, "", is_error=True, raw_response=getattr(e, 'raw_text', '')))
            break
        
        logger.info(f"LLM proposed action: {next_tool_name} with args: {next_tool_args}")
        
        # Enhanced loop detection - track current action
        current_action = f"{next_tool_name}:{json.dumps(next_tool_args, sort_keys=True)}"
        
        if current_action == last_action:
            consecutive_identical_count += 1
        else:
            consecutive_identical_count = 1
            last_action = current_action
        
        # Check for infinite loop (repetitive successful actions)
        if consecutive_identical_count > MAX_IDENTICAL_ACTIONS:
            logger.warning(f"Detected infinite loop: {consecutive_identical_count} identical actions")
            strategy_guidance = {
                "role": "user",
                "content": f"""🚨 INFINITE LOOP DETECTED! You've repeated the same action {consecutive_identical_count} times.

CRITICAL GUIDANCE:
- You're likely searching for file/directory paths in file contents (WRONG!)
- To find files/modules:
  * Use search_in_all_files_content to find class/function definitions first
  * Check existing import statements in related files
  * DO NOT blindly try to open files that may not exist
- Try a completely different approach
- Stop searching for the same term repeatedly

Current action: {next_tool_name}
Switch to a different strategy immediately!"""
            }
            messages.append(strategy_guidance)
            consecutive_identical_count = 0  # Reset counter
            continue  # Skip this tool execution
       
        try:
            tool_method = tool_manager.get_tool(next_tool_name)
            next_observation = tool_method(**next_tool_args) if next_tool_args else tool_method()
            
            logger.info(f"Tool observation (first 300 chars): {str(next_observation)[:300]}...")
            cot.add_action(EnhancedCOT.Action(next_thought, next_tool_name, next_tool_args, next_observation, is_error=False, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter))
            failure_tracker = {"tool_name": None, "args_hash": None, "count": 0}
            
            # Enhanced stuck detection - check for repeated patterns in recent successful actions
            if len(cot.thoughts) >= 5:
                recent = cot.thoughts[-5:]
                tool_names = [t.next_tool_name for t in recent]
                
                # Pattern 1: Same tool used 3+ times in last 5 steps
                from collections import Counter
                tool_counter = Counter(tool_names)
                most_common_tool, count = tool_counter.most_common(1)[0]
                
                if count >= 3 and next_tool_name == 'run_repo_tests':
                    # Check if tests are improving
                    test_runs = [t for t in recent if t.next_tool_name == 'run_repo_tests']
                    if len(test_runs) >= 2:
                        first_result = str(test_runs[0].observation)
                        last_result = str(test_runs[-1].observation)
                        first_passed = first_result.count('PASSED') + first_result.count('passed')
                        last_passed = last_result.count('PASSED') + last_result.count('passed')
                        
                        if last_passed <= first_passed:
                            logger.warning(f"[STUCK] Test results not improving after {len(test_runs)} runs")
                            is_stuck = True

        except Exception as e:
            error_msg = f"Tool Error: {e}"
            if not isinstance(e, EnhancedToolManager.Error):
                error_msg += f" {traceback.format_exc()}"
            
            logger.error(error_msg)
            cot.add_action(EnhancedCOT.Action(next_thought, next_tool_name, next_tool_args, error_msg, is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter))
            
            args_hash = hashlib.md5(str(sorted(next_tool_args.items())).encode()).hexdigest()
            if failure_tracker["tool_name"] == next_tool_name and failure_tracker["args_hash"] == args_hash:
                failure_tracker["count"] += 1
            else:
                failure_tracker = {"tool_name": next_tool_name, "args_hash": args_hash, "count": 1}
            
            if failure_tracker["count"] >= AgentConfig.MAX_CONSECUTIVE_TOOL_FAILURES:
                logger.warning(f"Tool '{next_tool_name}' has failed {failure_tracker['count']} consecutive times. Triggering stuck detection.")
                is_stuck = True
                
                # FIX 2: Complete Loop Detection - Add strategy guidance
                strategy_guidance = {
                    "role": "user", 
                    "content": f"🚨 STUCK DETECTED! Tool '{next_tool_name}' failed {failure_tracker['count']} consecutive times. Try a completely different approach! Stop using this tool and switch to a different strategy."
                }
                messages.append(strategy_guidance)
                is_stuck = False
                failure_tracker = {"tool_name": None, "args_hash": None, "count": 0}
            continue
        
        if step > 0 and (step % 5 == 0 or next_tool_name == "finish"):
            cot.save_state(AgentConfig.STATE_FILE_PATH)

        if next_tool_name == "finish":
            logger.info('Workflow finished successfully via "finish" tool.')
            break
    else:
        logger.warning(f"Workflow ended by reaching max steps ({n_max_steps}).")
        cot.add_action(EnhancedCOT.Action("max steps reached", "", {}, "", is_error=True))

    logger.info("Workflow execution complete. Generating final patch.")
    patch = tool_manager.get_final_git_patch()
    
    if os.path.exists(AgentConfig.STATE_FILE_PATH) and next_tool_name == "finish":
         os.remove(AgentConfig.STATE_FILE_PATH)

    return patch

