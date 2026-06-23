/**
 * ast_chunk.js
 *
 * Parses a JavaScript file into its Abstract Syntax Tree (AST) and
 * extracts each top-level function, class, and class method as its
 * own complete chunk -- instead of slicing the file by line count.
 *
 * This is called from Python (ingest.py) as a subprocess: we give it
 * a file path, it prints a JSON array of chunks to stdout.
 *
 * Usage: node src/ast_chunk.js /path/to/file.js
 */

const fs = require("fs");
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;

function chunkFile(filepath) {
  const code = fs.readFileSync(filepath, "utf-8");

  let ast;
  try {
    ast = parser.parse(code, {
      sourceType: "module",
      plugins: ["jsx", "typescript"],
      errorRecovery: true, // don't crash on minor syntax quirks
    });
  } catch (err) {
    // If a file genuinely can't be parsed, signal that clearly so
    // Python can fall back to line-based chunking for this file.
    console.error(`PARSE_ERROR: ${err.message}`);
    process.exit(1);
  }

  const chunks = [];
  const lines = code.split("\n");

  function extractChunk(node, label) {
    const startLine = node.loc.start.line;
    const endLine = node.loc.end.line;
    const text = lines.slice(startLine - 1, endLine).join("\n");

    // Skip tiny/trivial chunks (e.g. a one-line arrow function) --
    // not worth a whole embedding call for 2 lines of code.
    if (endLine - startLine < 2) return;

    chunks.push({
      text,
      start_line: startLine,
      end_line: endLine,
      label, // e.g. "function:createBudget" or "class:WellnessService"
    });
  }

  traverse(ast, {
    FunctionDeclaration(path) {
      const name = path.node.id ? path.node.id.name : "anonymous";
      extractChunk(path.node, `function:${name}`);
    },
    ClassDeclaration(path) {
      const name = path.node.id ? path.node.id.name : "anonymous";
      extractChunk(path.node, `class:${name}`);
    },
    ClassMethod(path) {
      const name = path.node.key.name || "anonymous";
      extractChunk(path.node, `method:${name}`);
    },
    // Catches `const createBudget = (req, res) => {...}` style functions,
    // which are extremely common in Express route handlers.
    VariableDeclarator(path) {
      const isFunctionLike =
        path.node.init &&
        (path.node.init.type === "ArrowFunctionExpression" ||
          path.node.init.type === "FunctionExpression");
      if (isFunctionLike && path.node.id.name) {
        extractChunk(path.node, `function:${path.node.id.name}`);
      }
    },
  });

  return chunks;
}

const filepath = process.argv[2];
if (!filepath) {
  console.error("Usage: node src/ast_chunk.js /path/to/file.js");
  process.exit(1);
}

const chunks = chunkFile(filepath);
console.log(JSON.stringify(chunks));
