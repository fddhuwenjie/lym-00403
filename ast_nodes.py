from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class SourceLocation:
    line: int
    column: int


@dataclass
class Type:
    kind: str
    name: str = ""
    base: Optional["Type"] = None
    params: Optional[List["Type"]] = None
    size: Optional[int] = None
    members: Optional[List["StructMember"]] = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.name:
            d["name"] = self.name
        if self.base is not None:
            d["base"] = self.base.to_dict()
        if self.params is not None:
            d["params"] = [p.to_dict() for p in self.params]
        if self.size is not None:
            d["size"] = self.size
        if self.members is not None:
            d["members"] = [m.to_dict() for m in self.members]
        return d


@dataclass
class StructMember:
    type: Type
    name: str

    def to_dict(self) -> dict:
        return {"type": self.type.to_dict(), "name": self.name}


@dataclass
class ASTNode:
    loc: SourceLocation

    def to_dict(self) -> dict:
        raise NotImplementedError


@dataclass
class Program(ASTNode):
    declarations: List[ASTNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "Program",
            "declarations": [d.to_dict() for d in self.declarations],
        }


@dataclass
class FunctionDef(ASTNode):
    return_type: Type = field(default_factory=lambda: Type("void"))
    name: str = ""
    params: List["ParamDecl"] = field(default_factory=list)
    body: Optional["Block"] = None

    def to_dict(self) -> dict:
        d: dict = {
            "kind": "FunctionDef",
            "return_type": self.return_type.to_dict(),
            "name": self.name,
            "params": [p.to_dict() for p in self.params],
        }
        if self.body is not None:
            d["body"] = self.body.to_dict()
        return d


@dataclass
class ParamDecl(ASTNode):
    type: Type = field(default_factory=lambda: Type("void"))
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": "ParamDecl",
            "type": self.type.to_dict(),
            "name": self.name,
        }


@dataclass
class StructDef(ASTNode):
    name: str = ""
    members: List[StructMember] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "StructDef",
            "name": self.name,
            "members": [m.to_dict() for m in self.members],
        }


@dataclass
class VarDecl(ASTNode):
    type: Type = field(default_factory=lambda: Type("void"))
    name: str = ""
    init: Optional[ASTNode] = None

    def to_dict(self) -> dict:
        d: dict = {
            "kind": "VarDecl",
            "type": self.type.to_dict(),
            "name": self.name,
        }
        if self.init is not None:
            d["init"] = self.init.to_dict()
        return d


@dataclass
class Block(ASTNode):
    stmts: List[ASTNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "Block",
            "stmts": [s.to_dict() for s in self.stmts],
        }


@dataclass
class ReturnStmt(ASTNode):
    value: Optional[ASTNode] = None

    def to_dict(self) -> dict:
        d: dict = {"kind": "ReturnStmt"}
        if self.value is not None:
            d["value"] = self.value.to_dict()
        return d


@dataclass
class IfStmt(ASTNode):
    condition: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))
    then_branch: ASTNode = field(default_factory=lambda: Block(SourceLocation(0, 0)))
    else_branch: Optional[ASTNode] = None

    def to_dict(self) -> dict:
        d: dict = {
            "kind": "IfStmt",
            "condition": self.condition.to_dict(),
            "then_branch": self.then_branch.to_dict(),
        }
        if self.else_branch is not None:
            d["else_branch"] = self.else_branch.to_dict()
        return d


@dataclass
class WhileStmt(ASTNode):
    condition: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))
    body: ASTNode = field(default_factory=lambda: Block(SourceLocation(0, 0)))

    def to_dict(self) -> dict:
        return {
            "kind": "WhileStmt",
            "condition": self.condition.to_dict(),
            "body": self.body.to_dict(),
        }


@dataclass
class ForStmt(ASTNode):
    init: Optional[ASTNode] = None
    condition: Optional[ASTNode] = None
    update: Optional[ASTNode] = None
    body: ASTNode = field(default_factory=lambda: Block(SourceLocation(0, 0)))

    def to_dict(self) -> dict:
        d: dict = {"kind": "ForStmt"}
        if self.init is not None:
            d["init"] = self.init.to_dict()
        if self.condition is not None:
            d["condition"] = self.condition.to_dict()
        if self.update is not None:
            d["update"] = self.update.to_dict()
        d["body"] = self.body.to_dict()
        return d


@dataclass
class BreakStmt(ASTNode):
    def to_dict(self) -> dict:
        return {"kind": "BreakStmt"}


@dataclass
class ContinueStmt(ASTNode):
    def to_dict(self) -> dict:
        return {"kind": "ContinueStmt"}


@dataclass
class ExprStmt(ASTNode):
    expr: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))

    def to_dict(self) -> dict:
        return {"kind": "ExprStmt", "expr": self.expr.to_dict()}


@dataclass
class BinaryExpr(ASTNode):
    op: str = ""
    left: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))
    right: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))

    def to_dict(self) -> dict:
        return {
            "kind": "BinaryExpr",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass
class UnaryExpr(ASTNode):
    op: str = ""
    operand: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))
    prefix: bool = True

    def to_dict(self) -> dict:
        return {
            "kind": "UnaryExpr",
            "op": self.op,
            "operand": self.operand.to_dict(),
            "prefix": self.prefix,
        }


@dataclass
class CallExpr(ASTNode):
    callee: ASTNode = field(default_factory=lambda: Identifier(SourceLocation(0, 0), ""))
    args: List[ASTNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "CallExpr",
            "callee": self.callee.to_dict(),
            "args": [a.to_dict() for a in self.args],
        }


@dataclass
class MemberExpr(ASTNode):
    obj: ASTNode = field(default_factory=lambda: Identifier(SourceLocation(0, 0), ""))
    member: str = ""
    is_arrow: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": "MemberExpr",
            "object": self.obj.to_dict(),
            "member": self.member,
            "access": "->" if self.is_arrow else ".",
        }


@dataclass
class IndexExpr(ASTNode):
    obj: ASTNode = field(default_factory=lambda: Identifier(SourceLocation(0, 0), ""))
    index: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))

    def to_dict(self) -> dict:
        return {
            "kind": "IndexExpr",
            "object": self.obj.to_dict(),
            "index": self.index.to_dict(),
        }


@dataclass
class AssignExpr(ASTNode):
    op: str = "="
    target: ASTNode = field(default_factory=lambda: Identifier(SourceLocation(0, 0), ""))
    value: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))

    def to_dict(self) -> dict:
        return {
            "kind": "AssignExpr",
            "op": self.op,
            "target": self.target.to_dict(),
            "value": self.value.to_dict(),
        }


@dataclass
class SizeofExpr(ASTNode):
    operand: Any = None

    def to_dict(self) -> dict:
        d: dict = {"kind": "SizeofExpr"}
        if self.operand is not None:
            if isinstance(self.operand, Type):
                d["operand"] = {"type": self.operand.to_dict()}
            else:
                d["operand"] = self.operand.to_dict()
        return d


@dataclass
class CastExpr(ASTNode):
    cast_type: Type = field(default_factory=lambda: Type("void"))
    expr: ASTNode = field(default_factory=lambda: IntLiteral(SourceLocation(0, 0), 0))

    def to_dict(self) -> dict:
        return {
            "kind": "CastExpr",
            "cast_type": self.cast_type.to_dict(),
            "expr": self.expr.to_dict(),
        }


@dataclass
class Identifier(ASTNode):
    name: str = ""

    def to_dict(self) -> dict:
        return {"kind": "Identifier", "name": self.name}


@dataclass
class IntLiteral(ASTNode):
    value: int = 0

    def to_dict(self) -> dict:
        return {"kind": "IntLiteral", "value": self.value}


@dataclass
class FloatLiteral(ASTNode):
    value: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": "FloatLiteral", "value": self.value}


@dataclass
class CharLiteral(ASTNode):
    value: str = ""

    def to_dict(self) -> dict:
        return {"kind": "CharLiteral", "value": self.value}


@dataclass
class StringLiteral(ASTNode):
    value: str = ""

    def to_dict(self) -> dict:
        return {"kind": "StringLiteral", "value": self.value}
