from typing import List, Optional, Set

from lexer import Token, TokenType, Lexer
from ast_nodes import (
    SourceLocation,
    Type,
    StructMember,
    ASTNode,
    Program,
    FunctionDef,
    ParamDecl,
    StructDef,
    VarDecl,
    Block,
    ReturnStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    BreakStmt,
    ContinueStmt,
    ExprStmt,
    BinaryExpr,
    UnaryExpr,
    CallExpr,
    MemberExpr,
    IndexExpr,
    AssignExpr,
    SizeofExpr,
    CastExpr,
    Identifier,
    IntLiteral,
    FloatLiteral,
    CharLiteral,
    StringLiteral,
)


class ParseError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Parse error at {line}:{column}: {message}")
        self.message = message
        self.line = line
        self.column = column


_TYPE_KEYWORDS: Set[TokenType] = {
    TokenType.KW_INT,
    TokenType.KW_CHAR,
    TokenType.KW_VOID,
    TokenType.KW_FLOAT,
    TokenType.KW_DOUBLE,
    TokenType.KW_SHORT,
    TokenType.KW_LONG,
    TokenType.KW_UNSIGNED,
    TokenType.KW_SIGNED,
    TokenType.KW_STRUCT,
    TokenType.KW_UNION,
    TokenType.KW_ENUM,
}

_ASSIGN_OPS: Set[TokenType] = {
    TokenType.ASSIGN,
    TokenType.PLUS_ASSIGN,
    TokenType.MINUS_ASSIGN,
    TokenType.STAR_ASSIGN,
    TokenType.SLASH_ASSIGN,
    TokenType.PERCENT_ASSIGN,
    TokenType.AMP_ASSIGN,
    TokenType.PIPE_ASSIGN,
    TokenType.CARET_ASSIGN,
    TokenType.LSHIFT_ASSIGN,
    TokenType.RSHIFT_ASSIGN,
}

_ASSIGN_OP_MAP = {
    TokenType.ASSIGN: "=",
    TokenType.PLUS_ASSIGN: "+=",
    TokenType.MINUS_ASSIGN: "-=",
    TokenType.STAR_ASSIGN: "*=",
    TokenType.SLASH_ASSIGN: "/=",
    TokenType.PERCENT_ASSIGN: "%=",
    TokenType.AMP_ASSIGN: "&=",
    TokenType.PIPE_ASSIGN: "|=",
    TokenType.CARET_ASSIGN: "^=",
    TokenType.LSHIFT_ASSIGN: "<<=",
    TokenType.RSHIFT_ASSIGN: ">>=",
}


class Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0
        self._errors: List[ParseError] = []

    @property
    def errors(self) -> List[ParseError]:
        return self._errors

    def _cur(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return self._tokens[-1]

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]

    def _advance(self) -> Token:
        tok = self._cur()
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._cur()
        if tok.type != ttype:
            self._error(f"expected {ttype.name}, got {tok.type.name} ('{tok.value}')")
            return tok
        return self._advance()

    def _match(self, ttype: TokenType) -> Optional[Token]:
        if self._cur().type == ttype:
            return self._advance()
        return None

    def _loc(self, tok: Optional[Token] = None) -> SourceLocation:
        t = tok or self._cur()
        return SourceLocation(t.line, t.column)

    def _error(self, msg: str) -> None:
        tok = self._cur()
        err = ParseError(msg, tok.line, tok.column)
        self._errors.append(err)

    def _synchronize(self) -> None:
        while self._cur().type != TokenType.EOF:
            if self._cur().type == TokenType.SEMI:
                self._advance()
                return
            if self._cur().type == TokenType.RBRACE:
                return
            self._advance()

    def parse(self) -> Program:
        decls: List[ASTNode] = []
        while self._cur().type != TokenType.EOF:
            try:
                decl = self._parse_top_level()
                if decl is not None:
                    decls.append(decl)
            except ParseError:
                self._synchronize()
        return Program(SourceLocation(1, 1), decls)

    def _parse_top_level(self) -> Optional[ASTNode]:
        if self._cur().type == TokenType.KW_TYPEDEF:
            return self._parse_typedef()

        is_struct = (
            self._cur().type == TokenType.KW_STRUCT
            and self._peek(1).type == TokenType.IDENT
            and self._peek(2).type == TokenType.LBRACE
        )
        if is_struct:
            return self._parse_struct_def()

        base_type = self._try_parse_type()
        if base_type is None:
            self._error(f"expected type specifier, got '{self._cur().value}'")
            self._synchronize()
            return None

        name_tok = self._expect(TokenType.IDENT)
        name = name_tok.value

        if self._cur().type == TokenType.LPAREN:
            return self._parse_function_def(base_type, name, name_tok)

        return self._parse_global_var_decl(base_type, name, name_tok)

    def _parse_typedef(self) -> Optional[ASTNode]:
        self._advance()
        base_type = self._try_parse_type()
        if base_type is None:
            self._error("expected type after typedef")
            self._synchronize()
            return None
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.SEMI)
        return VarDecl(
            self._loc(name_tok),
            type=Type("typedef", name=name_tok.value, base=base_type),
            name=name_tok.value,
        )

    def _parse_struct_def(self) -> StructDef:
        struct_tok = self._advance()
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.LBRACE)

        members: List[StructMember] = []
        while self._cur().type != TokenType.RBRACE and self._cur().type != TokenType.EOF:
            mtype = self._try_parse_type()
            if mtype is None:
                self._error("expected type in struct member")
                self._synchronize()
                break
            mname_tok = self._expect(TokenType.IDENT)
            mtype = self._maybe_parse_array_or_ptr(mtype)
            self._expect(TokenType.SEMI)
            members.append(StructMember(mtype, mname_tok.value))

        self._expect(TokenType.RBRACE)
        self._match(TokenType.SEMI)

        return StructDef(
            self._loc(struct_tok),
            name=name_tok.value,
            members=members,
        )

    def _parse_function_def(
        self, return_type: Type, name: str, name_tok: Token
    ) -> FunctionDef:
        self._expect(TokenType.LPAREN)
        params = self._parse_params()
        self._expect(TokenType.RPAREN)

        body = None
        if self._cur().type == TokenType.LBRACE:
            body = self._parse_block()

        return FunctionDef(
            self._loc(name_tok),
            return_type=return_type,
            name=name,
            params=params,
            body=body,
        )

    def _parse_params(self) -> List[ParamDecl]:
        params: List[ParamDecl] = []
        if self._cur().type == TokenType.RPAREN:
            return params

        if self._cur().type == TokenType.KW_VOID and self._peek(1).type == TokenType.RPAREN:
            self._advance()
            return params

        while True:
            ptype = self._try_parse_type()
            if ptype is None:
                self._error("expected parameter type")
                break
            pname = ""
            if self._cur().type == TokenType.IDENT:
                pname_tok = self._advance()
                pname = pname_tok.value
                ptype = self._maybe_parse_array_or_ptr(ptype)
            params.append(ParamDecl(SourceLocation(0, 0), type=ptype, name=pname))
            if not self._match(TokenType.COMMA):
                break

        return params

    def _parse_global_var_decl(
        self, base_type: Type, name: str, name_tok: Token
    ) -> VarDecl:
        full_type = self._maybe_parse_array_or_ptr(base_type)
        init = None
        if self._match(TokenType.ASSIGN):
            init = self._parse_expr()
        self._expect(TokenType.SEMI)
        return VarDecl(
            self._loc(name_tok),
            type=full_type,
            name=name,
            init=init,
        )

    def _try_parse_type(self) -> Optional[Type]:
        if self._cur().type in _TYPE_KEYWORDS:
            return self._parse_type_specifier()

        if self._cur().type == TokenType.IDENT:
            return Type("named", name=self._advance().value)

        return None

    def _parse_type_specifier(self) -> Type:
        is_const = False
        is_static = False
        is_unsigned = False
        base_names: List[str] = []

        while self._cur().type in (
            TokenType.KW_CONST,
            TokenType.KW_STATIC,
            TokenType.KW_UNSIGNED,
            TokenType.KW_SIGNED,
            TokenType.KW_SHORT,
            TokenType.KW_LONG,
        ):
            if self._cur().type == TokenType.KW_CONST:
                is_const = True
                self._advance()
            elif self._cur().type == TokenType.KW_STATIC:
                is_static = True
                self._advance()
            elif self._cur().type == TokenType.KW_UNSIGNED:
                is_unsigned = True
                self._advance()
            elif self._cur().type == TokenType.KW_SIGNED:
                self._advance()
            elif self._cur().type == TokenType.KW_SHORT:
                base_names.append("short")
                self._advance()
            elif self._cur().type == TokenType.KW_LONG:
                base_names.append("long")
                self._advance()

        result = None
        if self._cur().type in (TokenType.KW_INT, TokenType.KW_CHAR, TokenType.KW_VOID, TokenType.KW_FLOAT, TokenType.KW_DOUBLE):
            base_names.append(self._advance().value)
        elif self._cur().type == TokenType.KW_STRUCT:
            self._advance()
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                result = Type("struct", name=name)
            else:
                base_names.append("struct")

        elif self._cur().type == TokenType.KW_UNION:
            self._advance()
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                result = Type("union", name=name)
            else:
                base_names.append("union")

        elif self._cur().type == TokenType.KW_ENUM:
            self._advance()
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                result = Type("enum", name=name)
            else:
                base_names.append("enum")

        if result is None:
            if not base_names:
                base_names.append("int")

            combined = " ".join(base_names)
            result = Type("basic", name=combined)
            if is_unsigned:
                result = Type("unsigned", base=result)
            if is_const:
                result = Type("const", base=result)
            if is_static:
                result = Type("static", base=result)

        while self._cur().type == TokenType.STAR:
            self._advance()
            result = Type("pointer", base=result)

        return result

    def _maybe_parse_array_or_ptr(self, base: Type) -> Type:
        result = base
        if self._cur().type == TokenType.LBRACKET:
            self._advance()
            size: Optional[int] = None
            if self._cur().type != TokenType.RBRACKET:
                size_tok = self._expect(TokenType.INT_LIT)
                size = int(size_tok.value)
            self._expect(TokenType.RBRACKET)
            result = Type("array", base=result, size=size)
        return result

    def _parse_block(self) -> Block:
        brace_tok = self._expect(TokenType.LBRACE)
        stmts: List[ASTNode] = []
        while self._cur().type != TokenType.RBRACE and self._cur().type != TokenType.EOF:
            try:
                stmt = self._parse_stmt()
                if stmt is not None:
                    stmts.append(stmt)
            except ParseError:
                self._synchronize()
        self._expect(TokenType.RBRACE)
        return Block(self._loc(brace_tok), stmts=stmts)

    def _parse_stmt(self) -> Optional[ASTNode]:
        tok = self._cur()

        if tok.type == TokenType.LBRACE:
            return self._parse_block()

        if tok.type == TokenType.KW_RETURN:
            return self._parse_return()

        if tok.type == TokenType.KW_IF:
            return self._parse_if()

        if tok.type == TokenType.KW_WHILE:
            return self._parse_while()

        if tok.type == TokenType.KW_FOR:
            return self._parse_for()

        if tok.type == TokenType.KW_DO:
            return self._parse_do_while()

        if tok.type == TokenType.KW_BREAK:
            self._advance()
            self._expect(TokenType.SEMI)
            return BreakStmt(self._loc(tok))

        if tok.type == TokenType.KW_CONTINUE:
            self._advance()
            self._expect(TokenType.SEMI)
            return ContinueStmt(self._loc(tok))

        if self._is_type_start():
            return self._parse_local_var_decl()

        return self._parse_expr_stmt()

    def _is_type_start(self) -> bool:
        if self._cur().type in _TYPE_KEYWORDS:
            return True
        if self._cur().type in (TokenType.KW_CONST, TokenType.KW_STATIC):
            return True
        if self._cur().type == TokenType.IDENT:
            if self._peek(1).type == TokenType.IDENT:
                return True
            if self._peek(1).type == TokenType.STAR:
                idx = 2
                while self._peek(idx).type == TokenType.STAR:
                    idx += 1
                return self._peek(idx).type == TokenType.IDENT
        return False

    def _parse_local_var_decl(self) -> VarDecl:
        base_type = self._try_parse_type()
        if base_type is None:
            self._error("expected type in local declaration")
            self._synchronize()
            return VarDecl(SourceLocation(0, 0))
        name_tok = self._expect(TokenType.IDENT)
        full_type = self._maybe_parse_array_or_ptr(base_type)
        init = None
        if self._match(TokenType.ASSIGN):
            init = self._parse_assign_expr()
        self._expect(TokenType.SEMI)
        return VarDecl(
            self._loc(name_tok),
            type=full_type,
            name=name_tok.value,
            init=init,
        )

    def _parse_return(self) -> ReturnStmt:
        tok = self._advance()
        value = None
        if self._cur().type != TokenType.SEMI:
            value = self._parse_expr()
        self._expect(TokenType.SEMI)
        return ReturnStmt(self._loc(tok), value=value)

    def _parse_if(self) -> IfStmt:
        tok = self._advance()
        self._expect(TokenType.LPAREN)
        condition = self._parse_expr()
        self._expect(TokenType.RPAREN)
        then_branch = self._parse_stmt()
        else_branch = None
        if self._match(TokenType.KW_ELSE):
            else_branch = self._parse_stmt()
        return IfStmt(
            self._loc(tok),
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
        )

    def _parse_while(self) -> WhileStmt:
        tok = self._advance()
        self._expect(TokenType.LPAREN)
        condition = self._parse_expr()
        self._expect(TokenType.RPAREN)
        body = self._parse_stmt()
        return WhileStmt(self._loc(tok), condition=condition, body=body)

    def _parse_for(self) -> ForStmt:
        tok = self._advance()
        self._expect(TokenType.LPAREN)

        init = None
        if self._is_type_start():
            init = self._parse_local_var_decl()
        elif self._cur().type != TokenType.SEMI:
            init = self._parse_expr()
            self._expect(TokenType.SEMI)
        else:
            self._advance()

        condition = None
        if self._cur().type != TokenType.SEMI:
            condition = self._parse_expr()
        self._expect(TokenType.SEMI)

        update = None
        if self._cur().type != TokenType.RPAREN:
            update = self._parse_expr()
        self._expect(TokenType.RPAREN)

        body = self._parse_stmt()
        return ForStmt(
            self._loc(tok),
            init=init,
            condition=condition,
            update=update,
            body=body,
        )

    def _parse_do_while(self) -> WhileStmt:
        tok = self._advance()
        body = self._parse_stmt()
        self._expect(TokenType.KW_WHILE)
        self._expect(TokenType.LPAREN)
        condition = self._parse_expr()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.SEMI)
        return WhileStmt(self._loc(tok), condition=condition, body=body)

    def _parse_expr_stmt(self) -> ExprStmt:
        expr = self._parse_expr()
        self._expect(TokenType.SEMI)
        return ExprStmt(self._loc(), expr=expr)

    def _parse_expr(self) -> ASTNode:
        return self._parse_assign_expr()

    def _parse_assign_expr(self) -> ASTNode:
        left = self._parse_ternary()

        if self._cur().type in _ASSIGN_OPS:
            op_tok = self._advance()
            op = _ASSIGN_OP_MAP.get(op_tok.type, "=")
            right = self._parse_assign_expr()
            return AssignExpr(self._loc(op_tok), op=op, target=left, value=right)

        return left

    def _parse_ternary(self) -> ASTNode:
        cond = self._parse_or()

        if self._cur().type == TokenType.QUESTION:
            tok = self._advance()
            then_expr = self._parse_expr()
            self._expect(TokenType.COLON)
            else_expr = self._parse_ternary()
            return BinaryExpr(
                self._loc(tok),
                op="?:",
                left=BinaryExpr(self._loc(tok), op="?:cond", left=cond, right=then_expr),
                right=else_expr,
            )

        return cond

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._cur().type == TokenType.OR:
            tok = self._advance()
            right = self._parse_and()
            left = BinaryExpr(self._loc(tok), op="||", left=left, right=right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_bitor()
        while self._cur().type == TokenType.AND:
            tok = self._advance()
            right = self._parse_bitor()
            left = BinaryExpr(self._loc(tok), op="&&", left=left, right=right)
        return left

    def _parse_bitor(self) -> ASTNode:
        left = self._parse_bitxor()
        while self._cur().type == TokenType.PIPE:
            tok = self._advance()
            right = self._parse_bitxor()
            left = BinaryExpr(self._loc(tok), op="|", left=left, right=right)
        return left

    def _parse_bitxor(self) -> ASTNode:
        left = self._parse_bitand()
        while self._cur().type == TokenType.CARET:
            tok = self._advance()
            right = self._parse_bitand()
            left = BinaryExpr(self._loc(tok), op="^", left=left, right=right)
        return left

    def _parse_bitand(self) -> ASTNode:
        left = self._parse_equality()
        while self._cur().type == TokenType.AMP:
            tok = self._advance()
            right = self._parse_equality()
            left = BinaryExpr(self._loc(tok), op="&", left=left, right=right)
        return left

    def _parse_equality(self) -> ASTNode:
        left = self._parse_relational()
        while self._cur().type in (TokenType.EQ, TokenType.NEQ):
            tok = self._advance()
            op = "==" if tok.type == TokenType.EQ else "!="
            right = self._parse_relational()
            left = BinaryExpr(self._loc(tok), op=op, left=left, right=right)
        return left

    def _parse_relational(self) -> ASTNode:
        left = self._parse_shift()
        while self._cur().type in (TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            tok = self._advance()
            op_map = {
                TokenType.LT: "<",
                TokenType.GT: ">",
                TokenType.LE: "<=",
                TokenType.GE: ">=",
            }
            right = self._parse_shift()
            left = BinaryExpr(self._loc(tok), op=op_map[tok.type], left=left, right=right)
        return left

    def _parse_shift(self) -> ASTNode:
        left = self._parse_additive()
        while self._cur().type in (TokenType.LSHIFT, TokenType.RSHIFT):
            tok = self._advance()
            op = "<<" if tok.type == TokenType.LSHIFT else ">>"
            right = self._parse_additive()
            left = BinaryExpr(self._loc(tok), op=op, left=left, right=right)
        return left

    def _parse_additive(self) -> ASTNode:
        left = self._parse_multiplicative()
        while self._cur().type in (TokenType.PLUS, TokenType.MINUS):
            tok = self._advance()
            op = "+" if tok.type == TokenType.PLUS else "-"
            right = self._parse_multiplicative()
            left = BinaryExpr(self._loc(tok), op=op, left=left, right=right)
        return left

    def _parse_multiplicative(self) -> ASTNode:
        left = self._parse_unary()
        while self._cur().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            tok = self._advance()
            op_map = {
                TokenType.STAR: "*",
                TokenType.SLASH: "/",
                TokenType.PERCENT: "%",
            }
            right = self._parse_unary()
            left = BinaryExpr(self._loc(tok), op=op_map[tok.type], left=left, right=right)
        return left

    def _parse_unary(self) -> ASTNode:
        tok = self._cur()

        if tok.type == TokenType.MINUS:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="-", operand=operand, prefix=True)

        if tok.type == TokenType.PLUS:
            self._advance()
            return self._parse_unary()

        if tok.type == TokenType.NOT:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="!", operand=operand, prefix=True)

        if tok.type == TokenType.TILDE:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="~", operand=operand, prefix=True)

        if tok.type == TokenType.INC:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="++", operand=operand, prefix=True)

        if tok.type == TokenType.DEC:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="--", operand=operand, prefix=True)

        if tok.type == TokenType.STAR:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="*", operand=operand, prefix=True)

        if tok.type == TokenType.AMP:
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(self._loc(tok), op="&", operand=operand, prefix=True)

        if tok.type == TokenType.KW_SIZEOF:
            return self._parse_sizeof()

        if tok.type == TokenType.LPAREN:
            cast = self._try_parse_cast()
            if cast is not None:
                return cast

        return self._parse_postfix()

    def _parse_sizeof(self) -> SizeofExpr:
        tok = self._advance()
        if self._match(TokenType.LPAREN):
            maybe_type = self._try_parse_type()
            if maybe_type is not None:
                self._expect(TokenType.RPAREN)
                return SizeofExpr(self._loc(tok), operand=maybe_type)
            else:
                expr = self._parse_expr()
                self._expect(TokenType.RPAREN)
                return SizeofExpr(self._loc(tok), operand=expr)
        expr = self._parse_unary()
        return SizeofExpr(self._loc(tok), operand=expr)

    def _try_parse_cast(self) -> Optional[CastExpr]:
        save = self._pos
        try:
            self._advance()
            cast_type = self._try_parse_type()
            if cast_type is not None and self._cur().type == TokenType.RPAREN:
                self._advance()
                next_tok = self._cur()
                if next_tok.type in (
                    TokenType.IDENT,
                    TokenType.INT_LIT,
                    TokenType.FLOAT_LIT,
                    TokenType.CHAR_LIT,
                    TokenType.STRING_LIT,
                    TokenType.LPAREN,
                    TokenType.MINUS,
                    TokenType.PLUS,
                    TokenType.NOT,
                    TokenType.TILDE,
                    TokenType.STAR,
                    TokenType.AMP,
                    TokenType.INC,
                    TokenType.DEC,
                ):
                    expr = self._parse_unary()
                    return CastExpr(self._loc(), cast_type=cast_type, expr=expr)
            self._pos = save
            return None
        except ParseError:
            self._pos = save
            return None

    def _parse_postfix(self) -> ASTNode:
        expr = self._parse_primary()

        while True:
            if self._cur().type == TokenType.LPAREN:
                expr = self._parse_call(expr)
            elif self._cur().type == TokenType.LBRACKET:
                expr = self._parse_index(expr)
            elif self._cur().type == TokenType.DOT:
                tok = self._advance()
                member_tok = self._expect(TokenType.IDENT)
                expr = MemberExpr(
                    self._loc(tok), obj=expr, member=member_tok.value, is_arrow=False
                )
            elif self._cur().type == TokenType.ARROW:
                tok = self._advance()
                member_tok = self._expect(TokenType.IDENT)
                expr = MemberExpr(
                    self._loc(tok), obj=expr, member=member_tok.value, is_arrow=True
                )
            elif self._cur().type == TokenType.INC:
                tok = self._advance()
                expr = UnaryExpr(self._loc(tok), op="++", operand=expr, prefix=False)
            elif self._cur().type == TokenType.DEC:
                tok = self._advance()
                expr = UnaryExpr(self._loc(tok), op="--", operand=expr, prefix=False)
            else:
                break

        return expr

    def _parse_call(self, callee: ASTNode) -> CallExpr:
        tok = self._advance()
        args: List[ASTNode] = []
        if self._cur().type != TokenType.RPAREN:
            args.append(self._parse_assign_expr())
            while self._match(TokenType.COMMA):
                args.append(self._parse_assign_expr())
        self._expect(TokenType.RPAREN)
        return CallExpr(self._loc(tok), callee=callee, args=args)

    def _parse_index(self, obj: ASTNode) -> IndexExpr:
        tok = self._advance()
        index = self._parse_expr()
        self._expect(TokenType.RBRACKET)
        return IndexExpr(self._loc(tok), obj=obj, index=index)

    def _parse_primary(self) -> ASTNode:
        tok = self._cur()

        if tok.type == TokenType.INT_LIT:
            self._advance()
            value = int(tok.value, 0)
            return IntLiteral(self._loc(tok), value=value)

        if tok.type == TokenType.FLOAT_LIT:
            self._advance()
            return FloatLiteral(self._loc(tok), value=float(tok.value))

        if tok.type == TokenType.CHAR_LIT:
            self._advance()
            return CharLiteral(self._loc(tok), value=tok.value)

        if tok.type == TokenType.STRING_LIT:
            self._advance()
            return StringLiteral(self._loc(tok), value=tok.value)

        if tok.type == TokenType.IDENT:
            self._advance()
            return Identifier(self._loc(tok), name=tok.value)

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        self._error(f"unexpected token '{tok.value}'")
        self._advance()
        return IntLiteral(self._loc(tok), value=0)
