"""Synthetic database schema and data generator.

Generates SQLite databases with deterministic schemas and data
for SQL reasoning tasks.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnDef:
    """A column definition."""

    name: str
    col_type: str  # "INTEGER", "TEXT", "REAL"
    is_pk: bool = False
    fk_ref: str | None = None  # "table.column" for foreign keys


@dataclass
class TableDef:
    """A table definition."""

    name: str
    columns: list[ColumnDef]


@dataclass
class SchemaDefinition:
    """A complete database schema."""

    tables: list[TableDef]
    description: str = ""


@dataclass
class DatabaseInstance:
    """A generated database with schema and data."""

    schema: SchemaDefinition
    data: dict[str, list[dict[str, Any]]]  # table_name -> rows
    conn: sqlite3.Connection = field(repr=False, default=None)

    def __post_init__(self):
        if self.conn is None:
            self.conn = sqlite3.connect(":memory:")
            self._create_tables()
            self._insert_data()

    def _create_tables(self):
        """Create tables in SQLite."""
        for table in self.schema.tables:
            cols = []
            for col in table.columns:
                col_def = f"{col.name} {col.col_type}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                cols.append(col_def)
            sql = f"CREATE TABLE {table.name} ({', '.join(cols)})"
            self.conn.execute(sql)

    def _insert_data(self):
        """Insert data into tables."""
        for table in self.schema.tables:
            rows = self.data.get(table.name, [])
            if not rows:
                continue
            col_names = [c.name for c in table.columns]
            placeholders = ", ".join(["?"] * len(col_names))
            sql = f"INSERT INTO {table.name} ({', '.join(col_names)}) VALUES ({placeholders})"
            for row in rows:
                values = [row.get(c) for c in col_names]
                self.conn.execute(sql, values)
        self.conn.commit()

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as list of dicts."""
        cursor = self.conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_table_dump(self, table_name: str) -> list[tuple]:
        """Get all rows from a table as tuples."""
        cursor = self.conn.execute(f"SELECT * FROM {table_name}")
        return cursor.fetchall()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


# ---------------------------------------------------------------------------
# Schema families
# ---------------------------------------------------------------------------

# Name pools
EMPLOYEE_NAMES = [
    "Alice", "Bob", "Cara", "Dan", "Eva", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olga", "Paul",
]
DEPT_NAMES = ["Research", "Sales", "Engineering", "Marketing", "Finance", "HR"]
PROJECT_NAMES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
PRODUCT_NAMES = ["Widget", "Gadget", "Doohickey", "Thingamajig", "Whatchamacallit"]
CITIES = ["London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto"]


def build_employees_schema(
    rng: random.Random,
    num_tables: int = 2,
    schema_variant: str = "employees",
) -> tuple[SchemaDefinition, dict[str, list[dict[str, Any]]]]:
    """Build the 'employees' schema family with synthetic data."""

    tables: list[TableDef] = []
    data: dict[str, list[dict[str, Any]]] = {}

    # Departments table
    dept_table = TableDef(
        name="departments",
        columns=[
            ColumnDef("dept_id", "INTEGER", is_pk=True),
            ColumnDef("dept_name", "TEXT"),
        ],
    )
    tables.append(dept_table)
    num_depts = min(rng.randint(3, 5), len(DEPT_NAMES))
    dept_names = rng.sample(DEPT_NAMES, num_depts)
    dept_data = [
        {"dept_id": (i + 1) * 10, "dept_name": name}
        for i, name in enumerate(dept_names)
    ]
    data["departments"] = dept_data

    # Employees table
    if schema_variant == "employees_v2":
        # Schema drift: renamed columns
        emp_table = TableDef(
            name="employees",
            columns=[
                ColumnDef("emp_id", "INTEGER", is_pk=True),
                ColumnDef("full_name", "TEXT"),
                ColumnDef("department_id", "INTEGER", fk_ref="departments.dept_id"),
                ColumnDef("annual_salary", "INTEGER"),
                ColumnDef("hire_year", "INTEGER"),
            ],
        )
    elif schema_variant == "employees_v3":
        emp_table = TableDef(
            name="employees",
            columns=[
                ColumnDef("emp_id", "INTEGER", is_pk=True),
                ColumnDef("full_name", "TEXT"),
                ColumnDef("department_id", "INTEGER", fk_ref="departments.dept_id"),
                ColumnDef("base_salary", "INTEGER"),
                ColumnDef("bonus", "INTEGER"),
                ColumnDef("hire_year", "INTEGER"),
                ColumnDef("city", "TEXT"),
            ],
        )
    else:
        emp_table = TableDef(
            name="employees",
            columns=[
                ColumnDef("id", "INTEGER", is_pk=True),
                ColumnDef("name", "TEXT"),
                ColumnDef("dept_id", "INTEGER", fk_ref="departments.dept_id"),
                ColumnDef("salary", "INTEGER"),
            ],
        )
    tables.append(emp_table)

    # Generate employee data
    num_employees = rng.randint(5, 12)
    emp_names = rng.sample(EMPLOYEE_NAMES, min(num_employees, len(EMPLOYEE_NAMES)))
    emp_data = []
    for i, name in enumerate(emp_names):
        dept = rng.choice(dept_data)
        if schema_variant == "employees_v2":
            emp_data.append({
                "emp_id": i + 1,
                "full_name": name,
                "department_id": dept["dept_id"],
                "annual_salary": rng.randint(50, 200) * 1000,
                "hire_year": rng.randint(2015, 2025),
            })
        elif schema_variant == "employees_v3":
            salary = rng.randint(50, 200) * 1000
            emp_data.append({
                "emp_id": i + 1,
                "full_name": name,
                "department_id": dept["dept_id"],
                "base_salary": salary,
                "bonus": rng.randint(0, salary // 5),
                "hire_year": rng.randint(2015, 2025),
                "city": rng.choice(CITIES),
            })
        else:
            emp_data.append({
                "id": i + 1,
                "name": name,
                "dept_id": dept["dept_id"],
                "salary": rng.randint(50, 200) * 1000,
            })
    data["employees"] = emp_data

    # Optional: Projects table (for more complex queries)
    if num_tables >= 3:
        proj_table = TableDef(
            name="projects",
            columns=[
                ColumnDef("proj_id", "INTEGER", is_pk=True),
                ColumnDef("proj_name", "TEXT"),
                ColumnDef("dept_id", "INTEGER", fk_ref="departments.dept_id"),
                ColumnDef("budget", "INTEGER"),
            ],
        )
        tables.append(proj_table)
        num_projects = rng.randint(3, 6)
        proj_names = rng.sample(PROJECT_NAMES, min(num_projects, len(PROJECT_NAMES)))
        proj_data = [
            {
                "proj_id": i + 1,
                "proj_name": name,
                "dept_id": rng.choice(dept_data)["dept_id"],
                "budget": rng.randint(10, 500) * 1000,
            }
            for i, name in enumerate(proj_names)
        ]
        data["projects"] = proj_data

    # Optional: Assignments table (for join queries)
    if num_tables >= 4:
        assign_table = TableDef(
            name="assignments",
            columns=[
                ColumnDef("assign_id", "INTEGER", is_pk=True),
                ColumnDef("emp_id", "INTEGER", fk_ref="employees.id"),
                ColumnDef("proj_id", "INTEGER", fk_ref="projects.proj_id"),
                ColumnDef("role", "TEXT"),
            ],
        )
        tables.append(assign_table)
        roles = ["lead", "member", "advisor", "reviewer"]
        assign_data = []
        assign_id = 1
        for emp in emp_data:
            emp_id = emp.get("id") or emp.get("emp_id")
            if rng.random() < 0.7 and "projects" in data:
                proj = rng.choice(data["projects"])
                assign_data.append({
                    "assign_id": assign_id,
                    "emp_id": emp_id,
                    "proj_id": proj["proj_id"],
                    "role": rng.choice(roles),
                })
                assign_id += 1
        data["assignments"] = assign_data

    schema_def = SchemaDefinition(tables=tables, description=f"Schema family: {schema_variant}")
    return schema_def, data
