# Project Karthikeya — Codebase Auditor Report

This document details the architectural specifications and audit detection algorithms for the **Self-Healing Codebase Auditor** (`agent/auditor.py`) implemented inside Project Karthikeya.

---

## 1. Architectural Overview & Objectives

Codebases degrade over time as dependencies age, technical debt accumulates, and performance anti-patterns creep into source files. The Codebase Auditor provides automated static inspection to detect:
1. **Unused Dependencies**: Package declarations in `requirements.txt` that are never imported in source code.
2. **Security Vulnerabilities & CVEs**: Dependency version constraints matching known CVEs and weak cryptographic hash usage (`hashlib.md5`, `hashlib.sha1`).
3. **Performance Bottlenecks**: $O(N^2)$ nested loop structures, $N+1$ database query loops, and unclosed file handle resource leaks (`open()` without `with` context).
4. **Code Complexity**: Functions exceeding a Cyclomatic Complexity threshold ($V(G) > 10$).

---

## 2. Detection Algorithms & Methodology

### A. Unused Dependency Scanner
* **Methodology**: Crawls all `.py` files in the repository using Python `ast.walk` to construct a complete set of top-level imported module names.
* **Cross-Referencing**: Reads `requirements.txt`, strips version constraints (`==`, `>=`, `<=`), maps package names via `PACKAGE_IMPORT_MAP` (e.g. `gitpython` $\rightarrow$ `git`, `python-dotenv` $\rightarrow$ `dotenv`), and flags packages that have zero import references in the codebase.

### B. Security Vulnerability & CVE Scanner
* **Methodology**: Compares installed requirement versions against a database of known CVEs (`KNOWN_CVE_DATABASE`) covering packages such as `requests`, `urllib3`, `pyyaml`, `jinja2`, `pillow`, `cryptography`, and `setuptools`.
* **AST Code Security**: Inspects AST call nodes for weak hashing functions (`hashlib.md5()`, `hashlib.sha1()`) and flags recommendations to upgrade to `SHA-256` or `SHA-512`.

### C. Performance Bottleneck Analyzer
* **$O(N^2)$ Nested Loops**: Detects `ast.For` or `ast.While` loops containing nested child `ast.For`/`ast.While` loops. Recommends hash-map/dictionary index lookup optimizations.
* **$N+1$ Query Bottlenecks**: Detects database execution calls (`.execute()`, `.query()`, `.filter()`) invoked inside loop bodies. Recommends batching queries or utilizing JOINs.
* **Unclosed File Handlers**: Detects bare `open()` calls not enclosed inside an `ast.With` context manager.

### D. Cyclomatic Complexity Evaluator
* **Formula**: Computes Cyclomatic Complexity $V(G) = 1 + \pi$, where $\pi$ is the total count of decision points:
  $$\pi = \text{Count}(\text{If}) + \text{Count}(\text{For}) + \text{Count}(\text{While}) + \text{Count}(\text{Except}) + \text{Count}(\text{With}) + \text{Count}(\text{Assert}) + \text{Count}(\text{BoolOp Terms} - 1)$$
* **Threshold**: Functions with $V(G) > 10$ are flagged as high risk requiring modular decomposition.

---

## 3. Codebase Health Score Calculation

$$\text{Health Score} = \max\left(0.0, 100.0 - (\text{Total Issues Found} \times 5.0)\right)$$

---

## 4. Execution Interfaces

### CLI Command
```bash
python -m agent.cli --audit
```

### Dashboard API Endpoint
```http
GET /api/audit
```

---

## 5. Verification & Test Suite
The auditor engine is verified via `tests/test_auditor.py` with 100% green unit tests covering:
- Unused dependency detection (`test_audit_unused_dependencies`)
- Security CVE and weak hash detection (`test_audit_security_vulnerabilities`)
- $O(N^2)$ and $N+1$ query performance detection (`test_audit_performance_issues`)
- Cyclomatic complexity threshold evaluation (`test_audit_code_complexity`)
- Full report generation (`test_run_full_audit`)
