import json
import sys

from lexer import Lexer, LexerError
from parser import Parser, ParseError
from type_checker import TypeChecker
from codegen import generate_tac


def compile_source(source: str, check_types: bool = True) -> dict:
    lexer = Lexer(source)

    if lexer.errors:
        result: dict = {"ast": None, "lexer_errors": [], "parser_errors": [], "type_errors": []}
        for err in lexer.errors:
            result["lexer_errors"].append({
                "message": err.message,
                "line": err.line,
                "column": err.column,
            })
        return result

    parser = Parser(lexer.tokens)
    ast = parser.parse()

    result = {
        "ast": ast.to_dict(),
        "lexer_errors": [],
        "parser_errors": [],
        "type_errors": [],
    }

    for err in parser.errors:
        result["parser_errors"].append({
            "message": err.message,
            "line": err.line,
            "column": err.column,
        })

    if check_types and result["ast"] is not None:
        checker = TypeChecker()
        type_errors = checker.check(ast)
        for err in type_errors:
            result["type_errors"].append({
                "message": err.message,
                "line": err.line,
                "column": err.column,
            })

    return result


def main():
    args = sys.argv[1:]
    mode = "json"
    filename = None

    i = 0
    while i < len(args):
        if args[i] == "--tac":
            mode = "tac"
            i += 1
        else:
            filename = args[i]
            i += 1

    if filename is not None:
        with open(filename, "r") as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    if mode == "tac":
        lexer = Lexer(source)
        if lexer.errors:
            for err in lexer.errors:
                print(f"Lexer error at {err.line}:{err.column}: {err.message}", file=sys.stderr)
            sys.exit(1)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        if parser.errors:
            for err in parser.errors:
                print(f"Parse error at {err.line}:{err.column}: {err.message}", file=sys.stderr)
            sys.exit(1)
        tac_output = generate_tac(ast)
        print(tac_output)
    else:
        result = compile_source(source)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
