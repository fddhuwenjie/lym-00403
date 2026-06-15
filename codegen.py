from typing import List, Optional

from ast_nodes import (
    Program,
    FunctionDef,
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
    StructDef,
    ASTNode,
)


class TACInstruction:
    def __init__(self, op: str, result: str = "", arg1: str = "", arg2: str = "", label: str = ""):
        self.op = op
        self.result = result
        self.arg1 = arg1
        self.arg2 = arg2
        self.label = label

    def __str__(self) -> str:
        if self.op == "label":
            return f"{self.label}:"
        if self.op == "goto":
            return f"goto {self.label}"
        if self.op == "if_goto":
            return f"if {self.arg1} goto {self.label}"
        if self.op == "if_false_goto":
            return f"if_false {self.arg1} goto {self.label}"
        if self.op == "param":
            return f"param {self.arg1}"
        if self.op == "call":
            return f"call {self.arg1}, {self.arg2}"
        if self.op == "call_assign":
            return f"{self.result} = call {self.arg1}, {self.arg2}"
        if self.op == "return":
            if self.arg1:
                return f"return {self.arg1}"
            return "return"
        if self.op == "assign":
            return f"{self.result} = {self.arg1}"
        if self.op in ("+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "&&", "||", "&", "|", "^", "<<", ">>"):
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        if self.op == "uminus":
            return f"{self.result} = -{self.arg1}"
        if self.op == "unot":
            return f"{self.result} = !{self.arg1}"
        if self.op == "ubnot":
            return f"{self.result} = ~{self.arg1}"
        if self.op == "deref":
            return f"{self.result} = *{self.arg1}"
        if self.op == "addr":
            return f"{self.result} = &{self.arg1}"
        if self.op == "index":
            return f"{self.result} = {self.arg1}[{self.arg2}]"
        if self.op == "index_assign":
            return f"{self.arg1}[{self.arg2}] = {self.result}"
        if self.op == "member":
            return f"{self.result} = {self.arg1}.{self.arg2}"
        if self.op == "arrow":
            return f"{self.result} = {self.arg1}->{self.arg2}"
        if self.op == "cast":
            return f"{self.result} = ({self.arg2}){self.arg1}"
        if self.op == "pre_inc":
            return f"{self.result} = {self.arg1} + 1"
        if self.op == "pre_dec":
            return f"{self.result} = {self.arg1} - 1"
        if self.op == "post_inc":
            return f"{self.result} = {self.arg1} + 1"
        if self.op == "post_dec":
            return f"{self.result} = {self.arg1} - 1"
        return f"{self.result} = {self.arg1} {self.op} {self.arg2}"


class CodeGenerator:
    def __init__(self):
        self._instructions: List[TACInstruction] = []
        self._temp_count: int = 0
        self._label_count: int = 0
        self._break_labels: List[str] = []
        self._continue_labels: List[str] = []

    def _new_temp(self) -> str:
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def _new_label(self, hint: str = "L") -> str:
        name = f"{hint}{self._label_count}"
        self._label_count += 1
        return name

    def _emit(self, op: str, result: str = "", arg1: str = "", arg2: str = "", label: str = "") -> TACInstruction:
        instr = TACInstruction(op, result, arg1, arg2, label)
        self._instructions.append(instr)
        return instr

    def _emit_label(self, label: str) -> TACInstruction:
        return self._emit("label", label=label)

    def generate(self, program: Program) -> List[TACInstruction]:
        self._instructions = []
        self._temp_count = 0
        self._label_count = 0
        self._visit_program(program)
        return self._instructions

    def _visit_program(self, node: Program) -> None:
        for decl in node.declarations:
            if isinstance(decl, FunctionDef):
                self._visit_function_def(decl)
            elif isinstance(decl, VarDecl):
                self._visit_global_var_decl(decl)
            elif isinstance(decl, StructDef):
                pass

    def _visit_global_var_decl(self, node: VarDecl) -> None:
        if node.init is not None:
            val = self._visit_expr(node.init)
            self._emit("assign", result=node.name, arg1=val)
        else:
            self._emit("assign", result=node.name, arg1="0")

    def _visit_function_def(self, node: FunctionDef) -> None:
        self._emit_label(f"func_{node.name}")
        for param in node.params:
            if param.name:
                self._emit("param", arg1=param.name)
        if node.body is not None:
            self._visit_block(node.body)
        self._emit("return")

    def _visit_block(self, node: Block) -> None:
        for stmt in node.stmts:
            self._visit_stmt(stmt)

    def _visit_stmt(self, node: ASTNode) -> None:
        if isinstance(node, VarDecl):
            self._visit_var_decl(node)
        elif isinstance(node, Block):
            self._visit_block(node)
        elif isinstance(node, ReturnStmt):
            self._visit_return(node)
        elif isinstance(node, IfStmt):
            self._visit_if(node)
        elif isinstance(node, WhileStmt):
            self._visit_while(node)
        elif isinstance(node, ForStmt):
            self._visit_for(node)
        elif isinstance(node, BreakStmt):
            if self._break_labels:
                self._emit("goto", label=self._break_labels[-1])
        elif isinstance(node, ContinueStmt):
            if self._continue_labels:
                self._emit("goto", label=self._continue_labels[-1])
        elif isinstance(node, ExprStmt):
            self._visit_expr(node.expr)
        else:
            pass

    def _visit_var_decl(self, node: VarDecl) -> None:
        if node.init is not None:
            val = self._visit_expr(node.init)
            self._emit("assign", result=node.name, arg1=val)
        else:
            pass

    def _visit_return(self, node: ReturnStmt) -> None:
        if node.value is not None:
            val = self._visit_expr(node.value)
            self._emit("return", arg1=val)
        else:
            self._emit("return")

    def _visit_if(self, node: IfStmt) -> None:
        cond = self._visit_expr(node.condition)

        if node.else_branch is not None:
            else_label = self._new_label("else")
            end_label = self._new_label("endif")

            self._emit("if_false_goto", arg1=cond, label=else_label)
            self._visit_stmt(node.then_branch)
            self._emit("goto", label=end_label)
            self._emit_label(else_label)
            self._visit_stmt(node.else_branch)
            self._emit_label(end_label)
        else:
            end_label = self._new_label("endif")
            self._emit("if_false_goto", arg1=cond, label=end_label)
            self._visit_stmt(node.then_branch)
            self._emit_label(end_label)

    def _visit_while(self, node: WhileStmt) -> None:
        loop_label = self._new_label("while")
        end_label = self._new_label("endwhile")

        self._emit_label(loop_label)
        cond = self._visit_expr(node.condition)
        self._emit("if_false_goto", arg1=cond, label=end_label)

        self._break_labels.append(end_label)
        self._continue_labels.append(loop_label)
        self._visit_stmt(node.body)
        self._break_labels.pop()
        self._continue_labels.pop()

        self._emit("goto", label=loop_label)
        self._emit_label(end_label)

    def _visit_for(self, node: ForStmt) -> None:
        loop_label = self._new_label("for")
        body_label = self._new_label("forbody")
        end_label = self._new_label("endfor")

        if node.init is not None:
            if isinstance(node.init, VarDecl):
                self._visit_var_decl(node.init)
            else:
                self._visit_expr(node.init)

        self._emit_label(loop_label)

        if node.condition is not None:
            cond = self._visit_expr(node.condition)
            self._emit("if_false_goto", arg1=cond, label=end_label)

        self._break_labels.append(end_label)
        self._continue_labels.append(body_label)
        self._visit_stmt(node.body)
        self._break_labels.pop()
        self._continue_labels.pop()

        self._emit_label(body_label)
        if node.update is not None:
            self._visit_expr(node.update)

        self._emit("goto", label=loop_label)
        self._emit_label(end_label)

    def _visit_expr(self, node: ASTNode) -> str:
        if isinstance(node, IntLiteral):
            return str(node.value)
        elif isinstance(node, FloatLiteral):
            s = str(node.value)
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            return s
        elif isinstance(node, CharLiteral):
            return repr(node.value)
        elif isinstance(node, StringLiteral):
            return repr(node.value)
        elif isinstance(node, Identifier):
            return node.name
        elif isinstance(node, BinaryExpr):
            return self._visit_binary(node)
        elif isinstance(node, UnaryExpr):
            return self._visit_unary(node)
        elif isinstance(node, CallExpr):
            return self._visit_call(node)
        elif isinstance(node, MemberExpr):
            return self._visit_member(node)
        elif isinstance(node, IndexExpr):
            return self._visit_index(node)
        elif isinstance(node, AssignExpr):
            return self._visit_assign(node)
        elif isinstance(node, CastExpr):
            return self._visit_cast(node)
        elif isinstance(node, SizeofExpr):
            t = self._new_temp()
            self._emit("assign", result=t, arg1="sizeof")
            return t
        else:
            t = self._new_temp()
            self._emit("assign", result=t, arg1="0")
            return t

    def _visit_binary(self, node: BinaryExpr) -> str:
        if node.op == "?:":
            return self._visit_ternary(node)
        if node.op == "?:cond":
            return self._visit_expr(node.left)

        left = self._visit_expr(node.left)
        right = self._visit_expr(node.right)
        t = self._new_temp()

        if node.op in ("&&", "||"):
            return self._visit_short_circuit(node.op, left, right)

        self._emit(node.op, result=t, arg1=left, arg2=right)
        return t

    def _visit_short_circuit(self, op: str, left: str, right_expr: ASTNode) -> str:
        result = self._new_temp()
        self._emit("assign", result=result, arg1=left)
        skip_label = self._new_label("sc_skip")
        if op == "&&":
            self._emit("if_false_goto", arg1=result, label=skip_label)
        else:
            self._emit("if_goto", arg1=result, label=skip_label)
        right = self._visit_expr(right_expr)
        self._emit("assign", result=result, arg1=right)
        self._emit_label(skip_label)
        return result

    def _visit_ternary(self, node: BinaryExpr) -> str:
        cond_node = node.left
        cond_val = self._visit_expr(cond_node.left) if hasattr(cond_node, 'left') else self._visit_expr(cond_node)
        then_val_from_cond = self._visit_expr(cond_node.right) if hasattr(cond_node, 'right') else "0"

        result = self._new_temp()
        else_label = self._new_label("tern_else")
        end_label = self._new_label("tern_end")

        self._emit("if_false_goto", arg1=cond_val, label=else_label)
        then_val = self._visit_expr(node.left.right) if hasattr(node.left, 'right') else then_val_from_cond
        self._emit("assign", result=result, arg1=then_val)
        self._emit("goto", label=end_label)
        self._emit_label(else_label)
        else_val = self._visit_expr(node.right)
        self._emit("assign", result=result, arg1=else_val)
        self._emit_label(end_label)
        return result

    def _visit_unary(self, node: UnaryExpr) -> str:
        if node.op in ("++", "--") and node.prefix:
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            arith_op = "+" if node.op == "++" else "-"
            self._emit(arith_op, result=t, arg1=operand, arg2="1")
            self._emit("assign", result=operand, arg1=t)
            return t
        elif node.op in ("++", "--") and not node.prefix:
            operand = self._visit_expr(node.operand)
            old = self._new_temp()
            t = self._new_temp()
            self._emit("assign", result=old, arg1=operand)
            arith_op = "+" if node.op == "++" else "-"
            self._emit(arith_op, result=t, arg1=operand, arg2="1")
            self._emit("assign", result=operand, arg1=t)
            return old
        elif node.op == "-":
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            self._emit("uminus", result=t, arg1=operand)
            return t
        elif node.op == "!":
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            self._emit("unot", result=t, arg1=operand)
            return t
        elif node.op == "~":
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            self._emit("ubnot", result=t, arg1=operand)
            return t
        elif node.op == "*":
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            self._emit("deref", result=t, arg1=operand)
            return t
        elif node.op == "&":
            operand = self._visit_expr(node.operand)
            t = self._new_temp()
            self._emit("addr", result=t, arg1=operand)
            return t
        else:
            return self._visit_expr(node.operand)

    def _visit_call(self, node: CallExpr) -> str:
        for arg in node.args:
            arg_val = self._visit_expr(arg)
            self._emit("param", arg1=arg_val)

        if isinstance(node.callee, Identifier):
            func_name = node.callee.name
        else:
            func_name = self._visit_expr(node.callee)

        n_args = str(len(node.args))
        result = self._new_temp()
        self._emit("call_assign", result=result, arg1=func_name, arg2=n_args)
        return result

    def _visit_member(self, node: MemberExpr) -> str:
        obj = self._visit_expr(node.obj)
        t = self._new_temp()
        if node.is_arrow:
            self._emit("arrow", result=t, arg1=obj, arg2=node.member)
        else:
            self._emit("member", result=t, arg1=obj, arg2=node.member)
        return t

    def _visit_index(self, node: IndexExpr) -> str:
        obj = self._visit_expr(node.obj)
        idx = self._visit_expr(node.index)
        t = self._new_temp()
        self._emit("index", result=t, arg1=obj, arg2=idx)
        return t

    def _visit_assign(self, node: AssignExpr) -> str:
        val = self._visit_expr(node.value)

        if isinstance(node.target, Identifier):
            if node.op == "=":
                self._emit("assign", result=node.target.name, arg1=val)
                return node.target.name
            else:
                base_op = node.op[:-1]
                t = self._new_temp()
                self._emit(base_op, result=t, arg1=node.target.name, arg2=val)
                self._emit("assign", result=node.target.name, arg1=t)
                return node.target.name
        elif isinstance(node.target, IndexExpr):
            obj = self._visit_expr(node.target.obj)
            idx = self._visit_expr(node.target.index)
            if node.op == "=":
                self._emit("index_assign", result=val, arg1=obj, arg2=idx)
                return val
            else:
                base_op = node.op[:-1]
                t_old = self._new_temp()
                self._emit("index", result=t_old, arg1=obj, arg2=idx)
                t_new = self._new_temp()
                self._emit(base_op, result=t_new, arg1=t_old, arg2=val)
                self._emit("index_assign", result=t_new, arg1=obj, arg2=idx)
                return t_new
        elif isinstance(node.target, MemberExpr):
            obj = self._visit_expr(node.target.obj)
            if node.op == "=":
                if node.target.is_arrow:
                    self._emit("arrow", result=obj, arg2=node.target.member)
                else:
                    self._emit("member", result=obj, arg2=node.target.member)
                return val
            else:
                t = self._new_temp()
                if node.target.is_arrow:
                    self._emit("arrow", result=t, arg1=obj, arg2=node.target.member)
                else:
                    self._emit("member", result=t, arg1=obj, arg2=node.target.member)
                base_op = node.op[:-1]
                t2 = self._new_temp()
                self._emit(base_op, result=t2, arg1=t, arg2=val)
                if node.target.is_arrow:
                    self._emit("arrow", result=t2, arg1=obj, arg2=node.target.member)
                else:
                    self._emit("member", result=t2, arg1=obj, arg2=node.target.member)
                return t2
        else:
            t = self._new_temp()
            self._emit("assign", result=t, arg1=val)
            return t

    def _visit_cast(self, node: CastExpr) -> str:
        val = self._visit_expr(node.expr)
        t = self._new_temp()
        type_str = node.cast_type.name or node.cast_type.kind
        self._emit("cast", result=t, arg1=val, arg2=type_str)
        return t


def generate_tac(program: Program) -> str:
    gen = CodeGenerator()
    instructions = gen.generate(program)
    lines = []
    for instr in instructions:
        lines.append(str(instr))
    return "\n".join(lines)
