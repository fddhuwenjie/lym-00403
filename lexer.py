from dataclasses import dataclass
from typing import List
from enum import Enum, auto


class TokenType(Enum):
    INT_LIT = auto()
    FLOAT_LIT = auto()
    CHAR_LIT = auto()
    STRING_LIT = auto()
    IDENT = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    LSHIFT = auto()
    RSHIFT = auto()

    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto()
    SLASH_ASSIGN = auto()
    PERCENT_ASSIGN = auto()
    AMP_ASSIGN = auto()
    PIPE_ASSIGN = auto()
    CARET_ASSIGN = auto()
    LSHIFT_ASSIGN = auto()
    RSHIFT_ASSIGN = auto()

    INC = auto()
    DEC = auto()
    ARROW = auto()
    DOT = auto()

    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMI = auto()
    COMMA = auto()
    COLON = auto()
    QUESTION = auto()

    KW_INT = auto()
    KW_CHAR = auto()
    KW_VOID = auto()
    KW_FLOAT = auto()
    KW_DOUBLE = auto()
    KW_SHORT = auto()
    KW_LONG = auto()
    KW_UNSIGNED = auto()
    KW_SIGNED = auto()
    KW_CONST = auto()
    KW_STATIC = auto()
    KW_STRUCT = auto()
    KW_UNION = auto()
    KW_ENUM = auto()
    KW_TYPEDEF = auto()
    KW_RETURN = auto()
    KW_IF = auto()
    KW_ELSE = auto()
    KW_WHILE = auto()
    KW_FOR = auto()
    KW_DO = auto()
    KW_SWITCH = auto()
    KW_CASE = auto()
    KW_DEFAULT = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_SIZEOF = auto()

    EOF = auto()
    ERROR = auto()


KEYWORDS = {
    "int": TokenType.KW_INT,
    "char": TokenType.KW_CHAR,
    "void": TokenType.KW_VOID,
    "float": TokenType.KW_FLOAT,
    "double": TokenType.KW_DOUBLE,
    "short": TokenType.KW_SHORT,
    "long": TokenType.KW_LONG,
    "unsigned": TokenType.KW_UNSIGNED,
    "signed": TokenType.KW_SIGNED,
    "const": TokenType.KW_CONST,
    "static": TokenType.KW_STATIC,
    "struct": TokenType.KW_STRUCT,
    "union": TokenType.KW_UNION,
    "enum": TokenType.KW_ENUM,
    "typedef": TokenType.KW_TYPEDEF,
    "return": TokenType.KW_RETURN,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "while": TokenType.KW_WHILE,
    "for": TokenType.KW_FOR,
    "do": TokenType.KW_DO,
    "switch": TokenType.KW_SWITCH,
    "case": TokenType.KW_CASE,
    "default": TokenType.KW_DEFAULT,
    "break": TokenType.KW_BREAK,
    "continue": TokenType.KW_CONTINUE,
    "sizeof": TokenType.KW_SIZEOF,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class _State(Enum):
    START = auto()
    IDENT = auto()
    NUM_INT = auto()
    NUM_FRAC = auto()
    NUM_EXP = auto()
    NUM_EXP_SIGN = auto()
    NUM_EXP_DIGIT = auto()
    CHAR_START = auto()
    CHAR_BODY = auto()
    CHAR_ESCAPE = auto()
    STR_START = auto()
    STR_BODY = auto()
    STR_ESCAPE = auto()
    LINE_COMMENT = auto()
    BLOCK_COMMENT = auto()
    BLOCK_COMMENT_STAR = auto()
    OP = auto()
    DONE = auto()
    ERROR = auto()


_ESCAPE_MAP = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
}


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Lexer error at {line}:{column}: {message}")
        self.message = message
        self.line = line
        self.column = column


class Lexer:
    def __init__(self, source: str):
        self._src = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: List[Token] = []
        self._errors: List[LexerError] = []
        self._tokenize()

    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    @property
    def errors(self) -> List[LexerError]:
        return self._errors

    def _ch(self) -> str:
        if self._pos < len(self._src):
            return self._src[self._pos]
        return "\0"

    def _peek(self, offset: int = 1) -> str:
        idx = self._pos + offset
        if idx < len(self._src):
            return self._src[idx]
        return "\0"

    def _advance(self) -> str:
        ch = self._ch()
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _make_token(self, ttype: TokenType, value: str, line: int, col: int) -> Token:
        return Token(ttype, value, line, col)

    def _tokenize(self) -> None:
        state = _State.START
        buf = ""
        start_line = 1
        start_col = 1

        while state not in (_State.DONE, _State.ERROR):
            ch = self._ch()

            if state == _State.START:
                if ch == "\0":
                    self._tokens.append(
                        self._make_token(TokenType.EOF, "", self._line, self._col)
                    )
                    state = _State.DONE
                elif ch in " \t\r\n":
                    self._advance()
                elif ch == "/" and self._peek() == "/":
                    self._advance()
                    self._advance()
                    state = _State.LINE_COMMENT
                elif ch == "/" and self._peek() == "*":
                    self._advance()
                    self._advance()
                    state = _State.BLOCK_COMMENT
                elif ch.isalpha() or ch == "_":
                    start_line = self._line
                    start_col = self._col
                    buf = self._advance()
                    state = _State.IDENT
                elif ch.isdigit():
                    start_line = self._line
                    start_col = self._col
                    buf = self._advance()
                    state = _State.NUM_INT
                elif ch == "'":
                    start_line = self._line
                    start_col = self._col
                    self._advance()
                    buf = ""
                    state = _State.CHAR_START
                elif ch == '"':
                    start_line = self._line
                    start_col = self._col
                    self._advance()
                    buf = ""
                    state = _State.STR_START
                else:
                    start_line = self._line
                    start_col = self._col
                    buf = ""
                    state = _State.OP

            elif state == _State.IDENT:
                if ch.isalnum() or ch == "_":
                    buf += self._advance()
                else:
                    kw = KEYWORDS.get(buf)
                    if kw:
                        self._tokens.append(self._make_token(kw, buf, start_line, start_col))
                    else:
                        self._tokens.append(
                            self._make_token(TokenType.IDENT, buf, start_line, start_col)
                        )
                    buf = ""
                    state = _State.START

            elif state == _State.NUM_INT:
                if ch.isdigit():
                    buf += self._advance()
                elif ch == ".":
                    buf += self._advance()
                    state = _State.NUM_FRAC
                elif ch in "eE":
                    buf += self._advance()
                    state = _State.NUM_EXP_SIGN
                elif ch in "xX" and buf == "0":
                    buf += self._advance()
                    while self._ch() in "0123456789abcdefABCDEF":
                        buf += self._advance()
                    self._tokens.append(
                        self._make_token(TokenType.INT_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START
                else:
                    self._tokens.append(
                        self._make_token(TokenType.INT_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START

            elif state == _State.NUM_FRAC:
                if ch.isdigit():
                    buf += self._advance()
                elif ch in "eE":
                    buf += self._advance()
                    state = _State.NUM_EXP_SIGN
                else:
                    self._tokens.append(
                        self._make_token(TokenType.FLOAT_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START

            elif state == _State.NUM_EXP_SIGN:
                if ch in "+-":
                    buf += self._advance()
                    state = _State.NUM_EXP_DIGIT
                elif ch.isdigit():
                    state = _State.NUM_EXP_DIGIT
                else:
                    err = LexerError(
                        f"expected digit after exponent in number {buf}", start_line, start_col
                    )
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START

            elif state == _State.NUM_EXP_DIGIT:
                if ch.isdigit():
                    buf += self._advance()
                else:
                    self._tokens.append(
                        self._make_token(TokenType.FLOAT_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START

            elif state == _State.CHAR_START:
                if ch == "\0":
                    err = LexerError("unterminated character literal", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    state = _State.START
                elif ch == "\\":
                    self._advance()
                    state = _State.CHAR_ESCAPE
                elif ch == "'":
                    err = LexerError("empty character literal", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, "", start_line, start_col)
                    )
                    self._advance()
                    state = _State.START
                else:
                    buf = self._advance()
                    state = _State.CHAR_BODY

            elif state == _State.CHAR_BODY:
                if ch == "'":
                    self._advance()
                    self._tokens.append(
                        self._make_token(TokenType.CHAR_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START
                else:
                    err = LexerError(
                        f"multi-character character literal", start_line, start_col
                    )
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START

            elif state == _State.CHAR_ESCAPE:
                if ch == "\0":
                    err = LexerError("unterminated escape in char literal", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    state = _State.START
                else:
                    esc = self._advance()
                    buf = _ESCAPE_MAP.get(esc, esc)
                    state = _State.CHAR_BODY

            elif state == _State.STR_START:
                state = _State.STR_BODY

            elif state == _State.STR_BODY:
                if ch == "\0":
                    err = LexerError("unterminated string literal", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START
                elif ch == "\\":
                    self._advance()
                    state = _State.STR_ESCAPE
                elif ch == '"':
                    self._advance()
                    self._tokens.append(
                        self._make_token(TokenType.STRING_LIT, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START
                else:
                    buf += self._advance()

            elif state == _State.STR_ESCAPE:
                if ch == "\0":
                    err = LexerError("unterminated escape in string", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, buf, start_line, start_col)
                    )
                    buf = ""
                    state = _State.START
                else:
                    esc = self._advance()
                    buf += _ESCAPE_MAP.get(esc, esc)
                    state = _State.STR_BODY

            elif state == _State.LINE_COMMENT:
                if ch == "\0":
                    self._tokens.append(
                        self._make_token(TokenType.EOF, "", self._line, self._col)
                    )
                    state = _State.DONE
                elif ch == "\n":
                    self._advance()
                    state = _State.START
                else:
                    self._advance()

            elif state == _State.BLOCK_COMMENT:
                if ch == "\0":
                    err = LexerError("unterminated block comment", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, "/*", start_line, start_col)
                    )
                    state = _State.START
                elif ch == "*":
                    self._advance()
                    state = _State.BLOCK_COMMENT_STAR
                else:
                    self._advance()

            elif state == _State.BLOCK_COMMENT_STAR:
                if ch == "/":
                    self._advance()
                    state = _State.START
                elif ch == "*":
                    self._advance()
                elif ch == "\0":
                    err = LexerError("unterminated block comment", start_line, start_col)
                    self._errors.append(err)
                    self._tokens.append(
                        self._make_token(TokenType.ERROR, "/*", start_line, start_col)
                    )
                    state = _State.START
                else:
                    self._advance()
                    state = _State.BLOCK_COMMENT

            elif state == _State.OP:
                if ch == "\0":
                    self._resolve_op(buf, start_line, start_col)
                    self._tokens.append(
                        self._make_token(TokenType.EOF, "", self._line, self._col)
                    )
                    state = _State.DONE
                else:
                    candidate = buf + ch
                    if self._could_be_op(candidate):
                        buf += self._advance()
                    else:
                        self._resolve_op(buf, start_line, start_col)
                        buf = ""
                        state = _State.START

        for err in self._errors:
            pass

    @staticmethod
    def _could_be_op(s: str) -> bool:
        _OP_PREFIXES = {
            "+", "++", "+=",
            "-", "--", "-=", "->",
            "*", "*=",
            "/", "/=",
            "%", "%=",
            "&", "&&", "&=",
            "|", "||", "|=",
            "^", "^=",
            "~",
            "<<", "<<=",
            ">>", ">>=",
            "=", "==",
            "!", "!=",
            "<", "<=",
            ">", ">=",
            "(", ")", "{", "}", "[", "]",
            ";", ",", ".", ":", "?",
        }
        for op in _OP_PREFIXES:
            if op.startswith(s) and len(op) >= len(s):
                return True
        return False

    def _resolve_op(self, buf: str, line: int, col: int) -> None:
        _OP_MAP = {
            "+": TokenType.PLUS,
            "++": TokenType.INC,
            "+=": TokenType.PLUS_ASSIGN,
            "-": TokenType.MINUS,
            "--": TokenType.DEC,
            "-=": TokenType.MINUS_ASSIGN,
            "->": TokenType.ARROW,
            "*": TokenType.STAR,
            "*=": TokenType.STAR_ASSIGN,
            "/": TokenType.SLASH,
            "/=": TokenType.SLASH_ASSIGN,
            "%": TokenType.PERCENT,
            "%=": TokenType.PERCENT_ASSIGN,
            "&": TokenType.AMP,
            "&&": TokenType.AND,
            "&=": TokenType.AMP_ASSIGN,
            "|": TokenType.PIPE,
            "||": TokenType.OR,
            "|=": TokenType.PIPE_ASSIGN,
            "^": TokenType.CARET,
            "^=": TokenType.CARET_ASSIGN,
            "~": TokenType.TILDE,
            "<<": TokenType.LSHIFT,
            "<<=": TokenType.LSHIFT_ASSIGN,
            ">>": TokenType.RSHIFT,
            ">>=": TokenType.RSHIFT_ASSIGN,
            "=": TokenType.ASSIGN,
            "==": TokenType.EQ,
            "!": TokenType.NOT,
            "!=": TokenType.NEQ,
            "<": TokenType.LT,
            "<=": TokenType.LE,
            ">": TokenType.GT,
            ">=": TokenType.GE,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ";": TokenType.SEMI,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            ":": TokenType.COLON,
            "?": TokenType.QUESTION,
        }
        ttype = _OP_MAP.get(buf)
        if ttype is not None:
            self._tokens.append(self._make_token(ttype, buf, line, col))
        else:
            err = LexerError(f"unknown operator: {buf!r}", line, col)
            self._errors.append(err)
            self._tokens.append(self._make_token(TokenType.ERROR, buf, line, col))
