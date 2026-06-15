import json
import sys

from lexer import Lexer, LexerError
from parser import Parser, ParseError
from type_checker import TypeChecker


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
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    result = compile_source(source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
