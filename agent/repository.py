"""
Repository analyzer module.
Analyzes files, processes Python and JavaScript/TypeScript/other source code using AST,
builds dependency maps, identifies project architecture and tracks files affected by changes.
"""
import ast
import os
from typing import Any


class RepositoryAnalyzer:
    """
    Crawls local project paths, extracts architectural insights via AST parsing,
    finds dependencies, and keeps track of context and affected/related files.
    """
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)

    def scan_files(self) -> list[str]:
        """Lists all files in the repository, excluding standard ignore directories."""
        ignore_dirs = {".git", "node_modules", "venv", "__pycache__", "dist", "build", ".pytest_cache"}
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.root_dir)
                file_list.append(rel_path)
        return file_list

    def parse_python_ast(self, relative_path: str) -> dict[str, Any] | None:
        """
        Parses a python file to identify its classes, functions, and imports.
        """
        full_path = os.path.join(self.root_dir, relative_path)
        if not os.path.exists(full_path) or not relative_path.endswith(".py"):
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)

            classes = []
            functions = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return {
                "classes": classes,
                "functions": functions,
                "imports": imports
            }
        except Exception:
            return None

    def build_dependency_graph(self) -> dict[str, list[str]]:
        """
        Builds a basic dependency graph for python files in the repository based on imports.
        """
        graph: dict[str, list[str]] = {}
        files = self.scan_files()

        # Maps importable module strings back to python file names
        python_modules = {}
        for f in files:
            if f.endswith(".py"):
                mod_name = f.replace(".py", "").replace(os.sep, ".")
                python_modules[mod_name] = f

        for f in files:
            if f.endswith(".py"):
                analysis = self.parse_python_ast(f)
                if analysis:
                    deps = []
                    for imp in analysis["imports"]:
                        # Check if imported module is inside our repo
                        for mod in python_modules:
                            if imp == mod or imp.startswith(mod + "."):
                                deps.append(python_modules[mod])
                    graph[f] = list(set(deps))
        return graph

    def get_affected_files(self, modified_file: str) -> list[str]:
        """
        Traces downstream dependencies to see which other files might be affected by changes.
        """
        dep_graph = self.build_dependency_graph()
        affected = {modified_file}

        # Simple BFS / transitive closure over inverse dependency edges
        changed = True
        while changed:
            original_len = len(affected)
            for node, deps in dep_graph.items():
                if any(dep in affected for dep in deps):
                    affected.add(node)
            if len(affected) == original_len:
                changed = False

        return sorted(list(affected))

    def build_project_context(self) -> dict[str, Any]:
        """
        Returns full contextual analysis summarizing architecture and layout.
        """
        files = self.scan_files()
        python_files = [f for f in files if f.endswith(".py")]
        js_ts_files = [f for f in files if f.endswith((".js", ".jsx", ".ts", ".tsx"))]
        docker_files = [f for f in files if "dockerfile" in f.lower() or f.endswith(".dockerfile")]

        dep_graph = self.build_dependency_graph()

        return {
            "total_files": len(files),
            "python_files": len(python_files),
            "js_ts_files": len(js_ts_files),
            "docker_files": len(docker_files),
            "dependency_graph": dep_graph,
            "architecture_type": "Monorepo / Multi-layer" if len(js_ts_files) > 0 and len(python_files) > 0 else "Python project"
        }
