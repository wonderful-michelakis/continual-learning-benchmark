"""SQL reasoning task generator.

Generates questions over synthetic databases. The model must return the
query result (not the SQL query itself), which is verified by executing
a reference query against SQLite.
"""

from __future__ import annotations

import json
import random
from typing import Any

from continual_benchmark.core.constants import FamilyName, Split
from continual_benchmark.core.hashing import instance_uid
from continual_benchmark.core.schemas import Instance, Spec
from continual_benchmark.tasks.sql_reasoning.db_builder import (
    DatabaseInstance,
    build_employees_schema,
)


# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

class QueryTemplate:
    """A template for generating SQL questions + reference queries."""

    def __init__(self, question: str, sql: str, result_format: str = "list"):
        self.question = question
        self.sql = sql
        self.result_format = result_format  # "list", "scalar", "table"


def _get_query_templates(
    schema_variant: str,
    query_types: list[str],
    rng: random.Random,
    db: DatabaseInstance,
) -> list[QueryTemplate]:
    """Generate query templates appropriate for the schema and query types."""
    templates: list[QueryTemplate] = []

    # Determine column names based on schema variant
    if schema_variant in ("employees_v2", "employees_v3"):
        name_col = "full_name"
        dept_fk = "department_id"
        salary_col = "annual_salary" if schema_variant == "employees_v2" else "base_salary"
        emp_id_col = "emp_id"
    else:
        name_col = "name"
        dept_fk = "dept_id"
        salary_col = "salary"
        emp_id_col = "id"

    # Get actual data for realistic questions
    depts = db.data.get("departments", [])
    emps = db.data.get("employees", [])

    if not depts or not emps:
        return templates

    # Pick a department that has employees
    dept = rng.choice(depts)
    dept_name = dept["dept_name"]
    dept_id = dept["dept_id"]

    # Pick a salary threshold that gives non-trivial results
    salaries = [e[salary_col] for e in emps if salary_col in e]
    if salaries:
        salary_threshold = sorted(salaries)[len(salaries) // 2]
    else:
        salary_threshold = 100000

    if "select" in query_types:
        templates.append(QueryTemplate(
            question=f"What are the names of all employees?",
            sql=f"SELECT {name_col} FROM employees ORDER BY {name_col}",
            result_format="list",
        ))
        templates.append(QueryTemplate(
            question=f"What are the names of employees in the {dept_name} department?",
            sql=(
                f"SELECT e.{name_col} FROM employees e "
                f"JOIN departments d ON e.{dept_fk} = d.dept_id "
                f"WHERE d.dept_name = '{dept_name}' "
                f"ORDER BY e.{name_col}"
            ),
            result_format="list",
        ))

    if "filter" in query_types:
        templates.append(QueryTemplate(
            question=f"What are the names of employees with {salary_col.replace('_', ' ')} >= {salary_threshold}?",
            sql=(
                f"SELECT {name_col} FROM employees "
                f"WHERE {salary_col} >= {salary_threshold} "
                f"ORDER BY {name_col}"
            ),
            result_format="list",
        ))
        templates.append(QueryTemplate(
            question=(
                f"What are the names of employees in {dept_name} "
                f"with {salary_col.replace('_', ' ')} >= {salary_threshold}?"
            ),
            sql=(
                f"SELECT e.{name_col} FROM employees e "
                f"JOIN departments d ON e.{dept_fk} = d.dept_id "
                f"WHERE d.dept_name = '{dept_name}' AND e.{salary_col} >= {salary_threshold} "
                f"ORDER BY e.{name_col}"
            ),
            result_format="list",
        ))

    if "join" in query_types and "projects" in db.data:
        templates.append(QueryTemplate(
            question="For each department, list the department name and number of employees.",
            sql=(
                f"SELECT d.dept_name, COUNT(*) as emp_count "
                f"FROM departments d JOIN employees e ON d.dept_id = e.{dept_fk} "
                f"GROUP BY d.dept_name ORDER BY d.dept_name"
            ),
            result_format="table",
        ))

    if "aggregate" in query_types:
        templates.append(QueryTemplate(
            question=f"What is the average {salary_col.replace('_', ' ')} across all employees?",
            sql=f"SELECT ROUND(AVG({salary_col}), 2) as average FROM employees",
            result_format="scalar",
        ))
        templates.append(QueryTemplate(
            question=f"What is the highest {salary_col.replace('_', ' ')} in each department?",
            sql=(
                f"SELECT d.dept_name, MAX(e.{salary_col}) as max_salary "
                f"FROM employees e JOIN departments d ON e.{dept_fk} = d.dept_id "
                f"GROUP BY d.dept_name ORDER BY d.dept_name"
            ),
            result_format="table",
        ))

    if "subquery" in query_types:
        templates.append(QueryTemplate(
            question=(
                f"Which employees earn more than the average {salary_col.replace('_', ' ')}?"
            ),
            sql=(
                f"SELECT {name_col} FROM employees "
                f"WHERE {salary_col} > (SELECT AVG({salary_col}) FROM employees) "
                f"ORDER BY {name_col}"
            ),
            result_format="list",
        ))

    return templates


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_sql_instance(
    spec: Spec,
    stream_id: str,
    index: int,
    split: str,
    seed: int,
    rng: random.Random,
) -> Instance:
    """Generate a single SQL reasoning instance."""
    config = spec.generator_config
    schema_family = config.get("schema_family", "employees")
    query_types = config.get("query_types", ["select", "filter"])
    num_tables = spec.difficulty.num_tables or 2

    # Build database
    schema_def, data = build_employees_schema(rng, num_tables, schema_family)
    db = DatabaseInstance(schema=schema_def, data=data)

    try:
        # Get query templates and pick one
        templates = _get_query_templates(schema_family, query_types, rng, db)
        if not templates:
            # Fallback to a basic query
            templates = [QueryTemplate(
                question="What are the names of all employees?",
                sql="SELECT name FROM employees ORDER BY name",
                result_format="list",
            )]

        template = rng.choice(templates)

        # Execute reference query to get target
        try:
            raw_result = db.execute_query(template.sql)
        except Exception:
            # If query fails, use a simpler fallback
            raw_result = db.execute_query("SELECT name FROM employees LIMIT 5")
            template = QueryTemplate(
                question="List the first 5 employee names.",
                sql="SELECT name FROM employees LIMIT 5",
                result_format="list",
            )

        # Format result based on type
        if template.result_format == "list":
            if raw_result and len(raw_result[0]) == 1:
                key = list(raw_result[0].keys())[0]
                target_val = [row[key] for row in raw_result]
            else:
                target_val = raw_result
        elif template.result_format == "scalar":
            if raw_result:
                vals = list(raw_result[0].values())
                target_val = vals[0] if vals else None
            else:
                target_val = None
        else:  # "table"
            target_val = raw_result

        target = json.dumps(target_val, separators=(",", ":"))

        # Format prompt
        prompt = _format_sql_prompt(schema_def, data, template)

        # Metadata
        metadata: dict[str, Any] = {
            "difficulty": spec.difficulty.level,
            "num_tables": len(schema_def.tables),
            "query_type": template.result_format,
            "schema_family": schema_family,
            "drift_type": spec.drift.drift_type.value if spec.drift else None,
            "num_employees": len(data.get("employees", [])),
        }

        uid = instance_uid(
            spec.suite, FamilyName.SQL_REASONING.value,
            spec.stage, split, index, seed,
        )

        return Instance(
            uid=uid,
            suite=spec.suite,
            stream_id=stream_id,
            family=FamilyName.SQL_REASONING,
            spec_id=spec.spec_id,
            stage=spec.stage,
            split=Split(split),
            prompt=prompt,
            target=target,
            seed=seed,
            metadata=metadata,
        )
    finally:
        db.close()


def _format_sql_prompt(
    schema_def,
    data: dict[str, list[dict[str, Any]]],
    template: QueryTemplate,
) -> str:
    """Format a SQL reasoning prompt with schema and data."""
    lines = []

    # Schema description
    lines.append("Database schema:")
    lines.append("")
    for table in schema_def.tables:
        col_names = [c.name for c in table.columns]
        lines.append(f"{table.name}({', '.join(col_names)})")
    lines.append("")

    # Table contents
    for table in schema_def.tables:
        table_data = data.get(table.name, [])
        if not table_data:
            continue
        col_names = [c.name for c in table.columns]
        lines.append(f"{table.name}:")
        for row in table_data:
            values = [repr(row.get(c)) for c in col_names]
            lines.append(f"({', '.join(values)})")
        lines.append("")

    # Question
    lines.append(f"Question: {template.question}")
    lines.append("")

    # Return format
    if template.result_format == "list":
        lines.append("Return the answer as a JSON list of values.")
    elif template.result_format == "scalar":
        lines.append("Return the answer as a single JSON value.")
    else:
        lines.append("Return the answer as a JSON list of objects (one per row).")

    return "\n".join(lines)
