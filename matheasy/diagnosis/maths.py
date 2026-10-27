"""The maths layer: parse a student's line without simplifying it, check a step, find the wrong term.

    from matheasy.diagnosis import parse, equivalent, check_step, locate

    parse("1/2 + 1/3")                  # a Node tree, in the order the student wrote it
    equivalent("2(x + 3)", "2x + 6")    # True
    check_step("1/2 + 1/3", "2/5")      # False
    locate("1/2 + 1/3 + x", "2/5 + x")  # LocatedError(source='1/2 + 1/3', wrong='2/5', expected='5/6', ...)

SymPy's own parser reorders terms even with evaluate=False ("1/2 + 1/3" comes back
as "1/3 + 1/2") and flattens "2/3 ÷ 4/5", so a rule could not see what the student
wrote. We parse with Python's ast into a small tree that keeps order and grouping,
and use SymPy only to do the maths on it.

Supported input: integers, decimals, fractions, mixed numbers (3 1/2), percentages
(25%), + - × ÷ * / ^, brackets, letters as variables (2x, 3ab, 2(x + 1)), and one
= < > <= >= per line. A typed "a/b" between plain integers is a fraction, so
"2/3 ÷ 4/5" and "6 ÷ 2/3" mean what a student means. 25% is the number 1/4, so it
equals 0.25 but not 25. Exponents above MAX_EXPONENT and lines longer than
MAX_LENGTH are refused, so text from an LLM can never make equivalent() hang or crash.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import sympy


class ParseError(ValueError):
    """The line is not maths we can read. The orchestrator should ask the student to retype."""


# ---------------------------------------------------------------- the tree

class Node:
    """A parsed expression or relation. str(node) prints it back the way a student would write it."""

    def to_sympy(self) -> sympy.Expr:
        raise NotImplementedError

    def __str__(self) -> str:
        return _show(self)


@dataclass(frozen=True, eq=True)
class Num(Node):
    value: sympy.Rational
    text: str = field(compare=False)  # as written, e.g. "0.30"

    def to_sympy(self):
        return self.value


@dataclass(frozen=True)
class Mixed(Node):
    """A mixed number such as 3 1/2, kept as written: (whole, numerator, denominator)."""
    whole: int
    num: int
    den: int

    def to_sympy(self):
        return sympy.Integer(self.whole) + sympy.Rational(self.num, self.den)


@dataclass(frozen=True)
class Sym(Node):
    name: str

    def to_sympy(self):
        return sympy.Symbol(self.name)


@dataclass(frozen=True)
class Neg(Node):
    operand: Node

    def to_sympy(self):
        return -self.operand.to_sympy()


@dataclass(frozen=True)
class BinOp(Node):
    op: str  # + - * / ^
    left: Node
    right: Node

    def to_sympy(self):
        a, b = self.left.to_sympy(), self.right.to_sympy()
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        return a ** b


@dataclass(frozen=True)
class Rel(Node):
    op: str  # = < > <= >=
    left: Node
    right: Node

    def to_sympy(self):
        a, b = self.left.to_sympy(), self.right.to_sympy()
        return {"=": sympy.Eq, "<": sympy.Lt, ">": sympy.Gt, "<=": sympy.Le, ">=": sympy.Ge}[self.op](
            a, b, evaluate=False)


def as_fraction(node: Node) -> Optional[tuple[int, int]]:
    """(a, b) if node is a typed fraction a/b of whole numbers, else None. Not reduced: 2/4 -> (2, 4)."""
    if isinstance(node, BinOp) and node.op == "/" and _is_int(node.left) and _is_int(node.right):
        return int(node.left.value), int(node.right.value)
    return None


def _is_int(node: Node) -> bool:
    return isinstance(node, Num) and node.text.isdigit()


def _fraction_like(node: Node) -> bool:
    return as_fraction(node) is not None or isinstance(node, Mixed)


# ---------------------------------------------------------------- parsing

MAX_EXPONENT = 50
MAX_LENGTH = 300  # characters; a student's line is far shorter

_SYMBOLS = {"×": "*", "·": "*", "−": "-", "–": "-", "≤": "<=", "≥": ">=", "^": "**"}
# Mixed numbers and percentages become placeholder names (_m_3_1_2, _p_12_5) before
# the other rewrites, so those cannot split them; _convert() turns them back.
_MIXED_NUMBER = re.compile(r"(?<![\w.)/])(\d+)\s+(\d+)\s*/\s*(\d+)(?![\w.(])")
_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*%")
_OF = re.compile(r"(?<=[\d%])\s*\bof\b\s*")
_MIXED = re.compile(r"\d\s+\d")
_FUNCTION = re.compile(r"[A-Za-z]{2,}\s*\(")
_FRACTION = re.compile(r"(?<![\w.)])(\d+)\s*/\s*(\d+)(?![\w.(])")
_IMPLICIT = re.compile(r"(?<=[\d)])\s*(?=[A-Za-z(])|(?<=[A-Za-z])\s*(?=\()")
_EQUALS = re.compile(r"(?<![<>=!])=(?!=)")

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
_RELOPS = {ast.Eq: "=", ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">="}


def parse(text: Union[str, Node]) -> Node:
    """Parse one line of student maths without simplifying it. Raises ParseError."""
    if isinstance(text, Node):
        return text
    src = text.strip()
    if not src:
        raise ParseError("empty line")
    if len(src) > MAX_LENGTH:
        raise ParseError(f"line longer than {MAX_LENGTH} characters")
    if "_" in src:
        raise ParseError(f"cannot read {text!r}")
    for a, b in _SYMBOLS.items():
        src = src.replace(a, b)
    src = _OF.sub(" * ", src)  # 50% of 20, 1/2 of 12
    src = _MIXED_NUMBER.sub(r"_m_\1_\2_\3", src)
    src = _PERCENT.sub(lambda m: "_p_" + m.group(1).replace(".", "_"), src)
    if "%" in src:
        raise ParseError(f"cannot read {text!r}")
    if _MIXED.search(src):
        raise ParseError(f"two numbers with nothing between them: {text!r}")
    if _FUNCTION.search(src):
        raise ParseError(f"cannot read {text!r}")  # print(1), sqrt(4): not a product of letters
    src = _FRACTION.sub(r"(\1/\2)", src)  # before ÷, so "6 ÷ 2/3" is 6 ÷ (2/3)
    src = src.replace("÷", "/")
    src = _IMPLICIT.sub("*", src)
    src = _EQUALS.sub("==", src)
    try:
        tree = ast.parse(src, mode="eval")
        return _convert(tree.body, src, text)
    except (SyntaxError, RecursionError) as e:
        raise ParseError(f"cannot read {text!r}") from e


def _convert(n: ast.AST, src: str, original: str) -> Node:
    if isinstance(n, ast.Compare):
        if len(n.ops) != 1 or type(n.ops[0]) not in _RELOPS:
            raise ParseError(f"only one = < > <= >= per line: {original!r}")
        return Rel(_RELOPS[type(n.ops[0])], _convert(n.left, src, original),
                   _convert(n.comparators[0], src, original))
    if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
        node = BinOp(_BINOPS[type(n.op)], _convert(n.left, src, original), _convert(n.right, src, original))
        if node.op == "^":
            power = node.right.to_sympy()
            if power.is_number and abs(power) > MAX_EXPONENT:
                raise ParseError(f"exponent too large in {original!r}")
        return node
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return Neg(_convert(n.operand, src, original))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
        return _convert(n.operand, src, original)
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        written = ast.get_source_segment(src, n)
        return Num(sympy.Rational(written), written)
    if isinstance(n, ast.Name) and n.id.startswith("_m_"):
        whole, num, den = (int(part) for part in n.id[3:].split("_"))
        if den == 0:
            raise ParseError(f"zero denominator in {original!r}")
        return Mixed(whole, num, den)
    if isinstance(n, ast.Name) and n.id.startswith("_p_"):
        written = n.id[3:].replace("_", ".")
        return Num(sympy.Rational(written) / 100, f"{written}%")
    if isinstance(n, ast.Name) and n.id.isalpha() and n.id.isascii():
        node: Node = Sym(n.id[0])
        for letter in n.id[1:]:  # "ab" is a times b
            node = BinOp("*", node, Sym(letter))
        return node
    raise ParseError(f"cannot read {original!r}")


# ---------------------------------------------------------------- printing

def _prec(node: Node) -> int:
    if isinstance(node, Rel):
        return 0
    if isinstance(node, BinOp):
        if as_fraction(node):
            return 5
        return {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}[node.op]
    if isinstance(node, (Neg, Mixed)):
        return 3  # brackets when raised to a power: (3 1/2)^2
    return 5


def _show(node: Node) -> str:
    if isinstance(node, Num):
        return node.text
    if isinstance(node, Mixed):
        return f"{node.whole} {node.num}/{node.den}"
    if isinstance(node, Sym):
        return node.name
    if isinstance(node, Neg):
        inner = _show(node.operand)
        return f"-{inner}" if _prec(node.operand) > 3 else f"-({inner})"
    if isinstance(node, Rel):
        return f"{_show(node.left)} {node.op} {_show(node.right)}"
    if as_fraction(node):
        return f"{node.left.text}/{node.right.text}"
    p = _prec(node)
    left, right = _show(node.left), _show(node.right)
    if node.op == "^":
        left = left if _prec(node.left) > p else f"({left})"
        right = right if _prec(node.right) >= p else f"({right})"
        return f"{left}^{right}"
    left = left if _prec(node.left) >= p else f"({left})"
    right = right if _prec(node.right) > p else f"({right})"
    if node.op == "/":
        # a fraction divided by something reads as ÷, so it parses back the same way
        if _fraction_like(node.left) or _fraction_like(node.right):
            return f"{left} ÷ {right}"
        return f"{left}/{right}"
    if node.op == "*":
        # juxtapose only after a plain number, letter or bracket: never 1/2x, 3 1/2x or 25%x
        joinable = not _fraction_like(node.left) and (left[-1].isalnum() or left[-1] == ")")
        if joinable and _prec(node.right) >= p and _show(node.right)[0].isalpha():
            return f"{left}{_show(node.right)}"  # 2x, 3ab, 3x^2: no brackets needed, × is associative
        if joinable and right.startswith("("):
            return f"{left}{right}"  # 2(x + 3)
        return f"{left} × {right}"
    return f"{left} {node.op} {right}"


# ---------------------------------------------------------------- equivalence

def equivalent(a: Union[str, Node], b: Union[str, Node]) -> bool:
    """True if a and b are mathematically the same.

    Expressions: equal for every value of the variables (2(x + 3) and 2x + 6).
    Relations with variables: same solutions (x + 3 = 5 and x = 2).
    Relations between numbers: the same comparison of the same values (0.25 < 0.3 and 0.3 > 1/4).
    """
    a, b = parse(a), parse(b)
    if isinstance(a, Rel) or isinstance(b, Rel):
        return isinstance(a, Rel) and isinstance(b, Rel) and _same_relation(a, b)
    return _is_zero(a.to_sympy() - b.to_sympy())


def _is_zero(expr: sympy.Expr) -> bool:
    return sympy.simplify(expr) == 0


_FLIP = {">": "<", ">=": "<=", "<": "<", "<=": "<=", "=": "="}


def _canonical(rel: Rel) -> tuple[str, Node, Node]:
    if rel.op in (">", ">="):
        return _FLIP[rel.op], rel.right, rel.left
    return rel.op, rel.left, rel.right


def _same_relation(a: Rel, b: Rel) -> bool:
    op_a, la, ra = _canonical(a)
    op_b, lb, rb = _canonical(b)
    if op_a != op_b:
        return False
    da = la.to_sympy() - ra.to_sympy()
    db = lb.to_sympy() - rb.to_sympy()
    if not (da.free_symbols or db.free_symbols):
        same = equivalent(la, lb) and equivalent(ra, rb)
        return same or (op_a == "=" and equivalent(la, rb) and equivalent(ra, lb))
    if _is_zero(da) or _is_zero(db):
        return _is_zero(da) and _is_zero(db)
    ratio = sympy.simplify(da / db)
    if ratio.free_symbols or not ratio.is_finite or ratio == 0:
        return False
    return op_a == "=" or bool(ratio > 0)


def check_step(before: Union[str, Node], after: Union[str, Node]) -> bool:
    """True if going from `before` to `after` is mathematically valid (the two are equivalent)."""
    return equivalent(before, after)


# ---------------------------------------------------------------- locating the error

@dataclass(frozen=True)
class LocatedError:
    """Where a wrong step went wrong. A's build_query() turns this into the shortlister's query."""

    topic: str      # fractions | decimals | expressions, of the wrong part
    before: str     # the line before the step, as parsed
    after: str      # the student's line
    correct: str    # what the whole line should have become
    source: str     # the part of `before` the student changed
    wrong: str      # what the student wrote for that part
    expected: str   # what that part should have become
    difference: str  # wrong - expected
    source_node: Node = field(compare=False, repr=False)
    wrong_node: Node = field(compare=False, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        del d["source_node"], d["wrong_node"]
        return d


def locate(before: Union[str, Node], after: Union[str, Node]) -> Optional[LocatedError]:
    """Find the part of `before` the student got wrong in `after`. None if the step is correct.

    Terms the student copied unchanged (or changed correctly) are paired off; what is
    left on each side is the wrong part. Week-1 version: pairs terms of a sum, and
    the two sides of a relation. Anything else is treated as one changed part.
    """
    b, a = parse(before), parse(after)
    if equivalent(b, a):
        return None
    source, wrong = b, a
    if isinstance(b, Rel) and isinstance(a, Rel) and b.op == a.op:
        changed = [(sb, sa) for sb, sa in ((b.left, a.left), (b.right, a.right)) if not equivalent(sb, sa)]
        if len(changed) == 1:
            source, wrong = changed[0]
    if not isinstance(source, Rel) and not isinstance(wrong, Rel):
        source, wrong = _unmatched_terms(source, wrong)
    topic = topic_of(source, wrong)
    decimal = topic == "decimals"  # show 0.7, not 7/10
    if isinstance(source, Rel) or isinstance(wrong, Rel):
        expected, difference = _solved(source), ""
    else:
        value = sympy.simplify(source.to_sympy())
        expected = _sstr(value, decimal)
        difference = _sstr(sympy.simplify(wrong.to_sympy() - value), decimal)
    correct = _solved(b) if isinstance(b, Rel) else _sstr(sympy.simplify(b.to_sympy()), decimal)
    return LocatedError(
        topic=topic, before=str(b), after=str(a), correct=correct,
        source=str(source), wrong=str(wrong), expected=expected, difference=difference,
        source_node=source, wrong_node=wrong,
    )


def _sstr(expr: sympy.Expr, decimal: bool = False) -> str:
    """SymPy's text with ^ for powers; with decimal=True, a terminating fraction as a decimal (7/10 -> 0.7)."""
    if decimal and isinstance(expr, sympy.Rational) and not expr.is_integer:
        text = _as_decimal(int(expr.p), int(expr.q))
        if text is not None:
            return text
    return sympy.sstr(expr).replace("**", "^")


def _as_decimal(p: int, q: int) -> Optional[str]:
    """p/q written exactly as a decimal, or None if it does not terminate (1/3)."""
    rest, places = q, 0
    for factor in (2, 5):
        count = 0
        while rest % factor == 0:
            rest //= factor
            count += 1
        places = max(places, count)
    if rest != 1:
        return None
    digits = str(abs(p) * 10 ** places // q).rjust(places + 1, "0")
    whole, frac = digits[:-places], digits[-places:].rstrip("0")
    return f"{'-' if p < 0 else ''}{whole}.{frac}"


def _solved(node: Node) -> str:
    """An equation in one letter with one solution -> 'x = 2'; anything else is shown as written."""
    if isinstance(node, Rel) and node.op == "=":
        expr = node.left.to_sympy() - node.right.to_sympy()
        if len(expr.free_symbols) == 1:
            (x,) = expr.free_symbols
            solutions = sympy.solve(expr, x)
            if len(solutions) == 1:
                return f"{x} = {_sstr(solutions[0])}"
    return str(node)


def _terms(node: Node) -> list[tuple[int, Node]]:
    """Split a sum into signed terms, in the order written: a - b + c -> [(+1, a), (-1, b), (+1, c)]."""
    if isinstance(node, BinOp) and node.op in "+-":
        sign = 1 if node.op == "+" else -1
        return _terms(node.left) + [(s * sign, t) for s, t in _terms(node.right)]
    if isinstance(node, Neg):
        return [(-s, t) for s, t in _terms(node.operand)]
    return [(1, node)]


def _rebuild(terms: list[tuple[int, Node]]) -> Node:
    (sign, first), rest = terms[0], terms[1:]
    node = first if sign > 0 else Neg(first)
    for sign, t in rest:
        node = BinOp("+" if sign > 0 else "-", node, t)
    return node


def _unmatched_terms(b: Node, a: Node) -> tuple[Node, Node]:
    tb, ta = _terms(b), _terms(a)
    for same in (lambda x, y: x == y, lambda x, y: x[0] == y[0] and equivalent(x[1], y[1])):
        for t in list(ta):
            match = next((u for u in tb if same(u, t)), None)
            if match is not None and len(tb) > 1 and len(ta) > 1:
                tb.remove(match)
                ta.remove(t)
    return _rebuild(tb), _rebuild(ta)


def topic_of(*nodes: Node) -> str:
    """expressions if any letter appears, decimals if any decimal number does, else fractions."""
    found = set()

    def walk(n: Node):
        if isinstance(n, Sym):
            found.add("expressions")
        elif isinstance(n, Num) and ("." in n.text or "%" in n.text):  # percentages are in the decimals topic
            found.add("decimals")
        for child in (getattr(n, k, None) for k in ("operand", "left", "right")):
            if child is not None:
                walk(child)

    for n in nodes:
        walk(n)
    return "expressions" if "expressions" in found else "decimals" if "decimals" in found else "fractions"
