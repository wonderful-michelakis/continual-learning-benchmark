# Task Families

## DSL Execution

**What:** Execute programs in a small domain-specific language with stage-specific operator semantics.

**Prompt includes:** DSL rules, program, input bindings
**Output:** JSON with `result` key
**Verification:** Interpreter-based exact match after canonicalization

**Concept drift:** Operator semantics change (e.g., `ADD(x,y)` becomes `x+y+1`), new operators added.

**Difficulty controls:** Program length, nesting depth, operator variety, drift intensity.

## Structured Transformation

**What:** Transform JSON objects according to compositional rule chains.

**Prompt includes:** Transformation rules, input JSON
**Output:** Transformed JSON
**Verification:** Canonicalized JSON comparison (sorted keys, normalized numbers)

**Rule types:** Field rename, field removal, computed fields, list filtering, nested flattening, conditional transforms, aggregation, type casting.

**Concept drift:** Rule precedence changes, schema structure changes, new rule types.

## SQL Reasoning

**What:** Answer questions about synthetic relational databases.

**Prompt includes:** Schema definition, table contents, natural language question
**Output:** Query result as JSON (not SQL)
**Verification:** SQLite reference query execution + canonicalized comparison

**Concept drift:** Schema drift (renamed columns, new tables, changed join paths).

**Difficulty controls:** Number of tables, join depth, aggregation complexity, query types.

## API Code Generation

**What:** Write Python functions using a stage-specific toy API.

**Prompt includes:** API documentation, function signature, behavior requirements, example tests
**Output:** Python function code
**Verification:** Sandboxed test execution (subprocess + resource limits)

**Concept drift:** API behavior changes (e.g., `pairwise_sum` starts doubling values in v2.0), renamed functions, new constraints.

**Security:** Code execution uses subprocess isolation with AST pre-scanning and resource limits.
