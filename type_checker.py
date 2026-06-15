from typing import Dict, List, Optional

from ast_nodes import (
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


class TypeCheckError:
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column

    def __repr__(self) -> str:
        return f"TypeCheckError({self.line}:{self.column}: {self.message})"


ARITH_TYPES = {"int", "char", "float", "double", "short", "long", "long long", "short int", "long int", "long long int"}
INT_TYPES = {"int", "char", "short", "long", "long long", "short int", "long int", "long long int"}


def _unwrap(t: Type) -> Type:
    while t.kind in ("const", "static", "unsigned"):
        if t.base is not None:
            t = t.base
        else:
            break
    return t


def _type_name(t: Type) -> str:
    t = _unwrap(t)
    if t.kind == "basic":
        return t.name
    if t.kind == "pointer":
        return _type_name(t.base) + "*"
    if t.kind == "array":
        return _type_name(t.base) + "[]"
    if t.kind in ("struct", "union", "enum"):
        return f"{t.kind} {t.name}"
    if t.kind == "named":
        return t.name
    return t.kind


def _is_arithmetic(t: Type) -> bool:
    base = _unwrap(t)
    return base.kind == "basic" and base.name in ARITH_TYPES


def _is_integer(t: Type) -> bool:
    base = _unwrap(t)
    return base.kind == "basic" and base.name in INT_TYPES


def _is_pointer(t: Type) -> bool:
    return _unwrap(t).kind == "pointer"


def _is_scalar(t: Type) -> bool:
    return _is_arithmetic(t) or _is_pointer(t)


def _promote(left: Type, right: Type) -> Type:
    l = _unwrap(left)
    r = _unwrap(right)
    if l.kind == "basic" and l.name == "double":
        return Type("basic", name="double")
    if r.kind == "basic" and r.name == "double":
        return Type("basic", name="double")
    if l.kind == "basic" and l.name == "float":
        return Type("basic", name="float")
    if r.kind == "basic" and r.name == "float":
        return Type("basic", name="float")
    return Type("basic", name="int")


class SymbolTable:
    def __init__(self, parent: Optional["SymbolTable"] = None):
        self._parent = parent
        self._symbols: Dict[str, Type] = {}

    def define(self, name: str, t: Type) -> None:
        self._symbols[name] = t

    def lookup(self, name: str) -> Optional[Type]:
        if name in self._symbols:
            return self._symbols[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        return None

    def child(self) -> "SymbolTable":
        return SymbolTable(parent=self)


class TypeChecker:
    def __init__(self):
        self._errors: List[TypeCheckError] = []
        self._struct_defs: Dict[str, StructDef] = {}
        self._func_sigs: Dict[str, tuple] = {}
        self._current_func_return: Optional[Type] = None
        self._global_scope = SymbolTable()

    @property
    def errors(self) -> List[TypeCheckError]:
        return self._errors

    def _err(self, msg: str, node: ASTNode) -> None:
        self._errors.append(TypeCheckError(msg, node.loc.line, node.loc.column))

    def check(self, program: Program) -> List[TypeCheckError]:
        self._collect_structs(program)
        self._collect_funcs(program)
        for decl in program.declarations:
            self._check_top_level(decl)
        return self._errors

    def _collect_structs(self, program: Program) -> None:
        for decl in program.declarations:
            if isinstance(decl, StructDef):
                self._struct_defs[decl.name] = decl

    def _collect_funcs(self, program: Program) -> None:
        for decl in program.declarations:
            if isinstance(decl, FunctionDef):
                param_types = [p.type for p in decl.params]
                self._func_sigs[decl.name] = (decl.return_type, param_types)
                self._global_scope.define(decl.name, Type("function", name=decl.name))

    def _check_top_level(self, node: ASTNode) -> None:
        if isinstance(node, FunctionDef):
            self._check_function(node)
        elif isinstance(node, VarDecl):
            self._check_var_decl(node, self._global_scope)
        elif isinstance(node, StructDef):
            pass

    def _check_function(self, node: FunctionDef) -> None:
        scope = self._global_scope.child()
        self._current_func_return = node.return_type
        for param in node.params:
            scope.define(param.name, param.type)
        if node.body is not None:
            self._check_block(node.body, scope)

    def _check_block(self, node: Block, scope: SymbolTable) -> None:
        child = scope.child()
        for stmt in node.stmts:
            self._check_stmt(stmt, child)

    def _check_stmt(self, node: ASTNode, scope: SymbolTable) -> None:
        if isinstance(node, Block):
            self._check_block(node, scope)
        elif isinstance(node, VarDecl):
            self._check_var_decl(node, scope)
        elif isinstance(node, ReturnStmt):
            self._check_return(node, scope)
        elif isinstance(node, IfStmt):
            cond_type = self._check_expr(node.condition, scope)
            if cond_type is not None and not _is_scalar(cond_type):
                self._err("if condition must be scalar", node.condition)
            self._check_stmt(node.then_branch, scope)
            if node.else_branch is not None:
                self._check_stmt(node.else_branch, scope)
        elif isinstance(node, WhileStmt):
            cond_type = self._check_expr(node.condition, scope)
            if cond_type is not None and not _is_scalar(cond_type):
                self._err("while condition must be scalar", node.condition)
            self._check_stmt(node.body, scope)
        elif isinstance(node, ForStmt):
            for_scope = scope.child()
            if node.init is not None:
                if isinstance(node.init, VarDecl):
                    self._check_var_decl(node.init, for_scope)
                else:
                    self._check_expr(node.init, for_scope)
            if node.condition is not None:
                cond_type = self._check_expr(node.condition, for_scope)
                if cond_type is not None and not _is_scalar(cond_type):
                    self._err("for condition must be scalar", node.condition)
            if node.update is not None:
                self._check_expr(node.update, for_scope)
            self._check_stmt(node.body, for_scope)
        elif isinstance(node, ExprStmt):
            self._check_expr(node.expr, scope)
        elif isinstance(node, (BreakStmt, ContinueStmt)):
            pass

    def _check_var_decl(self, node: VarDecl, scope: SymbolTable) -> None:
        if node.init is not None:
            init_type = self._check_expr(node.init, scope)
            if init_type is not None:
                if not self._types_compatible(node.type, init_type):
                    self._err(
                        f"cannot initialize '{_type_name(node.type)}' with '{_type_name(init_type)}'",
                        node,
                    )
        scope.define(node.name, node.type)

    def _check_return(self, node: ReturnStmt, scope: SymbolTable) -> None:
        if node.value is not None:
            ret_type = self._check_expr(node.value, scope)
            if ret_type is not None and self._current_func_return is not None:
                unwrapped = _unwrap(self._current_func_return)
                if unwrapped.kind == "basic" and unwrapped.name == "void":
                    self._err("void function should not return a value", node)
                elif not self._types_compatible(self._current_func_return, ret_type):
                    self._err(
                        f"return type mismatch: expected '{_type_name(self._current_func_return)}', got '{_type_name(ret_type)}'",
                        node,
                    )
        else:
            if self._current_func_return is not None:
                unwrapped = _unwrap(self._current_func_return)
                if unwrapped.kind != "basic" or unwrapped.name != "void":
                    self._err("non-void function must return a value", node)

    def _check_expr(self, node: ASTNode, scope: SymbolTable) -> Optional[Type]:
        if isinstance(node, IntLiteral):
            return Type("basic", name="int")
        if isinstance(node, FloatLiteral):
            return Type("basic", name="double")
        if isinstance(node, CharLiteral):
            return Type("basic", name="char")
        if isinstance(node, StringLiteral):
            return Type("pointer", base=Type("basic", name="char"))
        if isinstance(node, Identifier):
            t = scope.lookup(node.name)
            if t is None:
                self._err(f"undeclared identifier '{node.name}'", node)
                return None
            return t
        if isinstance(node, BinaryExpr):
            return self._check_binary(node, scope)
        if isinstance(node, UnaryExpr):
            return self._check_unary(node, scope)
        if isinstance(node, CallExpr):
            return self._check_call(node, scope)
        if isinstance(node, MemberExpr):
            return self._check_member(node, scope)
        if isinstance(node, IndexExpr):
            return self._check_index(node, scope)
        if isinstance(node, AssignExpr):
            return self._check_assign(node, scope)
        if isinstance(node, SizeofExpr):
            return Type("basic", name="int")
        if isinstance(node, CastExpr):
            self._check_expr(node.expr, scope)
            return node.cast_type
        return None

    def _check_binary(self, node: BinaryExpr, scope: SymbolTable) -> Optional[Type]:
        left_type = self._check_expr(node.left, scope)
        right_type = self._check_expr(node.right, scope)
        if left_type is None or right_type is None:
            return None

        op = node.op
        if op in ("+", "-", "*", "/", "%"):
            if op in ("+", "-") and (_is_pointer(left_type) or _is_pointer(right_type)):
                if _is_pointer(left_type) and _is_integer(right_type):
                    return left_type
                if _is_pointer(right_type) and _is_integer(left_type) and op == "+":
                    return right_type
                if _is_pointer(left_type) and _is_pointer(right_type) and op == "-":
                    return Type("basic", name="int")
                self._err(f"invalid operands to '{op}'", node)
                return None
            if _is_arithmetic(left_type) and _is_arithmetic(right_type):
                return _promote(left_type, right_type)
            self._err(f"invalid operands to '{op}'", node)
            return None

        if op in ("<", ">", "<=", ">="):
            if _is_arithmetic(left_type) and _is_arithmetic(right_type):
                return Type("basic", name="int")
            if _is_pointer(left_type) and _is_pointer(right_type):
                return Type("basic", name="int")
            self._err(f"invalid operands to '{op}'", node)
            return None

        if op in ("==", "!="):
            if _is_arithmetic(left_type) and _is_arithmetic(right_type):
                return Type("basic", name="int")
            if _is_pointer(left_type) and _is_pointer(right_type):
                return Type("basic", name="int")
            if self._types_compatible(left_type, right_type):
                return Type("basic", name="int")
            self._err(f"invalid operands to '{op}'", node)
            return None

        if op in ("&&", "||"):
            if _is_scalar(left_type) and _is_scalar(right_type):
                return Type("basic", name="int")
            self._err(f"invalid operands to '{op}'", node)
            return None

        if op in ("&", "|", "^"):
            if _is_integer(left_type) and _is_integer(right_type):
                return _promote(left_type, right_type)
            self._err(f"invalid operands to '{op}'", node)
            return None

        if op in ("<<", ">>"):
            if _is_integer(left_type) and _is_integer(right_type):
                return left_type
            self._err(f"invalid operands to '{op}'", node)
            return None

        return None

    def _check_unary(self, node: UnaryExpr, scope: SymbolTable) -> Optional[Type]:
        operand_type = self._check_expr(node.operand, scope)
        if operand_type is None:
            return None

        op = node.op
        if op == "-":
            if _is_arithmetic(operand_type):
                return operand_type
            self._err(f"invalid operand to unary '-'", node)
            return None
        if op == "!":
            if _is_scalar(operand_type):
                return Type("basic", name="int")
            self._err(f"invalid operand to '!'", node)
            return None
        if op == "~":
            if _is_integer(operand_type):
                return operand_type
            self._err(f"invalid operand to '~'", node)
            return None
        if op in ("++", "--"):
            if _is_arithmetic(operand_type) or _is_pointer(operand_type):
                return operand_type
            self._err(f"invalid operand to '{op}'", node)
            return None
        if op == "*":
            unwrapped = _unwrap(operand_type)
            if unwrapped.kind == "pointer" and unwrapped.base is not None:
                return unwrapped.base
            if unwrapped.kind == "array" and unwrapped.base is not None:
                return unwrapped.base
            self._err(f"cannot dereference non-pointer type '{_type_name(operand_type)}'", node)
            return None
        if op == "&":
            return Type("pointer", base=operand_type)

        return operand_type

    def _check_call(self, node: CallExpr, scope: SymbolTable) -> Optional[Type]:
        callee_type = self._check_expr(node.callee, scope)
        arg_types = []
        for arg in node.args:
            at = self._check_expr(arg, scope)
            if at is not None:
                arg_types.append(at)

        if callee_type is None:
            return None

        if isinstance(node.callee, Identifier):
            name = node.callee.name
            if name in self._func_sigs:
                ret_type, param_types = self._func_sigs[name]
                if len(arg_types) != len(param_types):
                    self._err(
                        f"function '{name}' expects {len(param_types)} arguments, got {len(arg_types)}",
                        node,
                    )
                else:
                    for i, (expected, actual) in enumerate(zip(param_types, arg_types)):
                        if not self._types_compatible(expected, actual):
                            self._err(
                                f"argument {i+1} of '{name}': expected '{_type_name(expected)}', got '{_type_name(actual)}'",
                                node,
                            )
                return ret_type

        return Type("basic", name="int")

    def _check_member(self, node: MemberExpr, scope: SymbolTable) -> Optional[Type]:
        obj_type = self._check_expr(node.obj, scope)
        if obj_type is None:
            return None

        check_type = obj_type
        if node.is_arrow:
            unwrapped = _unwrap(obj_type)
            if unwrapped.kind == "pointer" and unwrapped.base is not None:
                check_type = unwrapped.base
            else:
                self._err(
                    f"'->' requires pointer to struct/union, got '{_type_name(obj_type)}'",
                    node,
                )
                return None

        unwrapped = _unwrap(check_type)
        struct_name = ""
        if unwrapped.kind == "struct":
            struct_name = unwrapped.name
        elif unwrapped.kind == "union":
            struct_name = unwrapped.name
        else:
            self._err(
                f"member access requires struct/union type, got '{_type_name(check_type)}'",
                node,
            )
            return None

        sdef = self._struct_defs.get(struct_name)
        if sdef is None:
            self._err(f"undefined struct/union '{struct_name}'", node)
            return None

        for member in sdef.members:
            if member.name == node.member:
                return member.type

        self._err(f"no member '{node.member}' in '{struct_name}'", node)
        return None

    def _check_index(self, node: IndexExpr, scope: SymbolTable) -> Optional[Type]:
        obj_type = self._check_expr(node.obj, scope)
        idx_type = self._check_expr(node.index, scope)

        if obj_type is None:
            return None

        unwrapped = _unwrap(obj_type)
        if unwrapped.kind == "pointer" and unwrapped.base is not None:
            return unwrapped.base
        if unwrapped.kind == "array" and unwrapped.base is not None:
            return unwrapped.base

        self._err(f"cannot index non-array/non-pointer type '{_type_name(obj_type)}'", node)
        return None

    def _check_assign(self, node: AssignExpr, scope: SymbolTable) -> Optional[Type]:
        target_type = self._check_expr(node.target, scope)
        value_type = self._check_expr(node.value, scope)

        if target_type is None or value_type is None:
            return target_type

        if node.op != "=":
            effective_op = node.op[:-1]
            if effective_op in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"):
                if not self._types_compatible(target_type, value_type):
                    self._err(
                        f"compound assignment type mismatch: '{_type_name(target_type)}' and '{_type_name(value_type)}'",
                        node,
                    )

        if not self._types_compatible(target_type, value_type):
            self._err(
                f"assignment type mismatch: cannot assign '{_type_name(value_type)}' to '{_type_name(target_type)}'",
                node,
            )

        return target_type

    def _types_compatible(self, target: Type, source: Type) -> bool:
        t = _unwrap(target)
        s = _unwrap(source)

        if t.kind == "basic" and s.kind == "basic":
            return True

        if t.kind == "pointer" and s.kind == "pointer":
            return True

        if t.kind == "pointer" and s.kind == "basic" and s.name == "int":
            return True

        if t.kind == "basic" and t.name == "int" and s.kind == "pointer":
            return True

        if t.kind in ("struct", "union") and s.kind in ("struct", "union"):
            return t.name == s.name

        if t.kind == "array" and s.kind == "array":
            if t.base is not None and s.base is not None:
                return self._types_compatible(t.base, s.base)

        if t.kind == "pointer" and s.kind == "array" and s.base is not None:
            return self._types_compatible(t.base, s.base)

        return False
