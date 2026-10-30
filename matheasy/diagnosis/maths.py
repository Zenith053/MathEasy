"""The maths layer: parse a student's line without simplifying it, check a step, find the wrong term.

    from matheasy.diagnosis import parse, equivalent, check_step, locate

    parse("1/2 + 1/3")                  # a Node tree, in the order the student wrote it
    equivalent("2(x + 3)", "2x + 6")    # True
    check_step("1/2 + 1/3", "2/5")      # False
    locate("1/2 + 1/3 + x", "2/5 + x")  # LocatedError(source='1/2 + 1/3', wrong='2/5', expected='5/6', ...)

SymPy's own parser reorders terms even with evaluate=False ("1/2 + 1/3" comes back
as "1/3 + 1/2") and flattens "2/3 ÷ 4/5", so a rule could not see what the student
wrote. We parse with our own small Pratt parser into a tree that keeps order and
grouping, and use SymPy only to do the maths on it.

Supported input: integers, decimals, fractions, mixed numbers (3 1/2), percentages
(25%), + - × ÷ * / ^, brackets, letters as variables (2x, 3ab, 2(x + 1)), and one
= < > <= >= per line. A typed "a/b" between plain integers is a fraction, so
"2/3 ÷ 4/5" and "6 ÷ 2/3" mean what a student means. 25% is the number 1/4, so it
equals 0.25 but not 25. Exponents above MAX_EXPONENT and lines longer than
MAX_LENGTH are refused, so text from an LLM can never make equivalent() hang or crash.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
class Word(Node):
    """A named quantity written as a word: Area (a known quantity word), or {rate} (marked)."""
    name: str                         # lower case, so Area and area are one quantity
    text: str = field(compare=False)  # as written

    def to_sympy(self):
        return sympy.Symbol(self.name)  # never clashes with a letter: a word has 2+ letters


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
    typed_fraction: bool = field(default=False, compare=False, repr=False)  # 3/4 typed as one number, not 3 ÷ 4

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
#
# A Pratt parser: each operator has a binding power, and an operand belongs to
# whichever neighbouring operator binds tighter. Student notation is handled where
# it is read, not by rewriting the text first:
#
#   = < > <= >=                   relation, lowest, at most one per line
#   + -                    20     left to right
#   × ÷ * /  "of"  2x 2(x+1)  30  left to right; "of" and writing side by side mean ×
#   unary - +              40     -x^2 is -(x^2), as in maths
#   ^ **                   50     right to left: 2^3^2 is 2^(9)
#   a/b of two whole numbers      read as one number (a fraction) before any operator,
#   whole + fraction (3 1/2)      so 2/3 ÷ 4/5 and 6 ÷ 2/3 mean what a student means
#   25%                           a percentage is read with its number: 25% is 1/4
#   π                             the constant π (22/7 and 3.14 are ordinary numbers)
#   Area, {rate}                  a known quantity word, or any word in braces, is one
#                                 quantity; other letters written together multiply

MAX_EXPONENT = 50
MAX_LENGTH = 300  # characters; a student's line is far shorter
MAX_DEPTH = 100   # nested brackets and signs
MIN_RUN = 4       # unknown runs this long or longer may be a word: xyzw or rate? (see notation_issue)

QUANTITY_WORDS_FILE = Path(__file__).with_name("quantity_words.txt")
QUANTITY_WORDS = frozenset(
    w.strip().lower() for w in QUANTITY_WORDS_FILE.read_text().splitlines()
    if w.strip() and not w.lstrip().startswith("#")
)

_INFIX = {"+": (20, 21), "-": (20, 21), "*": (30, 31), "/": (30, 31), "÷": (30, 31), "^": (50, 50)}
_TIMES = (30, 31)    # "of", and writing side by side: 2x, 2(x + 1)
_LETTERS = (35, 36)  # letters side by side are one term: a/bc is a/(bc), but xy^2 is x × y^2
_UNARY = 40

_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<num>\d+(?:\.\d*)?|\.\d+)(?P<pct>\s*%)?
  | (?P<brace>\{\s*(?P<braced>[A-Za-z]{2,})\s*\})
  | (?P<pi>π)
  | (?P<let>[A-Za-z]+)
  | (?P<op>\*\*|<=|>=|==|[-+*/÷^()=<>×·−–≤≥])
""", re.VERBOSE)
_OP_NAMES = {"**": "^", "×": "*", "·": "*", "−": "-", "–": "-", "==": "=", "≤": "<=", "≥": ">="}
_RELATIONS = {"=", "<", ">", "<=", ">="}


@dataclass(frozen=True)
class _Token:
    kind: str     # num | pct | let (one letter, or π) | word | op | rel
    text: str     # as written (num, pct, let, word) or the operator's name (op, rel)
    pos: int      # where it starts in the line
    spaced: bool  # whitespace before it


def _tokenize(line: str) -> list[_Token]:
    tokens, pos, spaced = [], 0, False
    while pos < len(line):
        m = _TOKEN.match(line, pos)
        if m is None:
            raise ParseError(f"unexpected {line[pos]!r} at character {pos + 1} in {line!r}")
        if m["space"]:
            spaced = True
        elif m["num"]:
            kind, text = ("pct", m["num"] + "%") if m["pct"] else ("num", m["num"])
            tokens.append(_Token(kind, text, pos, spaced))
        elif m["brace"]:  # {rate}: a word marked as one quantity
            if line[m.end():].lstrip().startswith("("):
                raise ParseError(f"cannot read {m['brace']}(...) at character {pos + 1} in {line!r}")
            tokens.append(_Token("word", m["braced"], pos, spaced))
        elif m["pi"]:
            tokens.append(_Token("let", "π", pos, spaced))  # stands like a letter: 2πr
        elif m["let"] == "of" and tokens and tokens[-1].kind in ("num", "pct"):
            tokens.append(_Token("op", "*", pos, spaced))  # 50% of 20, 1/2 of 12
        elif m["let"]:
            word = m["let"]
            if len(word) > 1 and line[m.end():].lstrip().startswith("("):
                raise ParseError(f"cannot read {word}(...) at character {pos + 1} in {line!r}")  # sqrt(4)
            if word.lower() in QUANTITY_WORDS:  # Area: one quantity
                tokens.append(_Token("word", word, pos, spaced))
            else:
                for k, letter in enumerate(word):  # one token per letter, so xy^2 is x × y^2
                    tokens.append(_Token("let", letter, pos + k, spaced if k == 0 else False))
        else:
            name = _OP_NAMES.get(m["op"], m["op"])
            tokens.append(_Token("rel" if name in _RELATIONS else "op", name, pos, spaced))
        if not m["space"]:
            spaced = False
        pos = m.end()
    return tokens


def parse(text: Union[str, Node]) -> Node:
    """Parse one line of student maths without simplifying it. Raises ParseError."""
    if isinstance(text, Node):
        return text
    line = text.strip()
    if not line:
        raise ParseError("empty line")
    if len(line) > MAX_LENGTH:
        raise ParseError(f"line longer than {MAX_LENGTH} characters")
    try:
        return _Parser(line).line()
    except RecursionError as e:  # a safety net; MAX_DEPTH stops this first
        raise ParseError(f"cannot read {line!r}") from e


class _Parser:
    def __init__(self, line: str):
        self.line_text = line
        self.tokens = _tokenize(line)
        self.i = 0
        self.depth = 0

    # -- token helpers

    def peek(self, ahead: int = 0) -> Optional[_Token]:
        j = self.i + ahead
        return self.tokens[j] if j < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def previous(self) -> _Token:
        return self.tokens[self.i - 1]

    def error(self, reason: str, token: Optional[_Token] = None) -> ParseError:
        where = f"at character {token.pos + 1}" if token else "at the end"
        return ParseError(f"{reason} {where} in {self.line_text!r}")

    def is_whole(self, ahead: int) -> bool:
        token = self.peek(ahead)
        return token is not None and token.kind == "num" and token.text.isdigit()

    def is_op(self, ahead: int, op: str) -> bool:
        token = self.peek(ahead)
        return token is not None and token.kind == "op" and token.text == op

    def glued(self, ahead: int) -> bool:
        """The token is followed, with no space, by a letter or a bracket: 1/2x, 3 1/2(."""
        token = self.peek(ahead)
        return (token is not None and not token.spaced
                and (token.kind == "let" or (token.kind == "op" and token.text == "(")))

    def mixed_at(self, ahead: int) -> bool:
        """A mixed number starts here: whole, space, whole / whole (3 1/2).

        Not straight after a "/" with no space (6/3 1/2 is two numbers), and a mixed
        number wins over a fraction: in 1 / 3 1/2 the 3 belongs to 3 1/2.
        """
        start = self.peek(ahead)
        top = self.peek(ahead + 1)
        if not (self.is_whole(ahead) and self.is_whole(ahead + 1) and top.spaced
                and self.is_op(ahead + 2, "/") and self.is_whole(ahead + 3) and not self.glued(ahead + 4)):
            return False
        j = self.i + ahead
        after_slash = j > 0 and self.tokens[j - 1].kind == "op" and self.tokens[j - 1].text == "/"
        return not (after_slash and not start.spaced)

    # -- grammar

    def line(self) -> Node:
        left = self.expression(0)
        token = self.peek()
        if token is not None and token.kind == "rel":
            self.take()
            right = self.expression(0)
            extra = self.peek()
            if extra is not None and extra.kind == "rel":
                raise self.error("only one = < > <= >= per line", extra)
            left = Rel(token.text, left, right)
        token = self.peek()
        if token is not None:
            raise self.error(f"unexpected {token.text!r}", token)
        return left

    def expression(self, min_power: int) -> Node:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise self.error("brackets or signs nested too deeply")
        left = self.operand()
        while (token := self.peek()) is not None:
            if token.kind == "op" and token.text in _INFIX:
                left_power, right_power = _INFIX[token.text]
                if left_power < min_power:
                    break
                self.take()
                left = self.binop(token, left, self.expression(right_power))
            elif (powers := self.side_by_side(token)) is not None:
                if powers[0] < min_power:
                    break
                left = BinOp("*", left, self.expression(powers[1]))
            elif token.kind in ("num", "pct") and self.previous().kind in ("num", "pct"):
                raise self.error("two numbers with nothing between them", token)
            else:
                break
        self.depth -= 1
        return left

    def side_by_side(self, token: _Token) -> Optional[tuple[int, int]]:
        """Writing side by side means times: 2x, xy, 2(x + 1), (x + 1)(x + 2), x(y + 1).

        Returns the binding powers, or None if the two cannot stand side by side.
        """
        before = self.previous()
        opens = token.kind == "op" and token.text == "("
        after_number = before.kind == "num" or (before.kind == "op" and before.text == ")")
        if token.kind == "word":  # 2Area, (n + 1){cost}; but a word never runs into letters
            return _TIMES if after_number else None
        if before.kind == "word":  # Area x, Area(…): write Area × x
            return None
        if token.kind != "let" and not opens:
            return None
        if before.kind == "num" or (before.kind == "op" and before.text == ")"):
            return _TIMES
        if before.kind == "let" and token.kind == "let":
            return None if token.spaced else _LETTERS  # xy, but "x y" is two letters with nothing between
        return _TIMES if before.kind == "let" and opens else None

    def binop(self, token: _Token, left: Node, right: Node) -> Node:
        op = "/" if token.text == "÷" else token.text
        node = BinOp(op, left, right)
        if op == "^":
            power = right.to_sympy()
            if power.is_number and abs(power) > MAX_EXPONENT:
                raise self.error("exponent too large", token)
        return node

    def operand(self) -> Node:
        token = self.peek()
        if token is None:
            raise self.error("expected a number, letter or bracket")
        if token.kind == "op" and token.text in "-+":
            self.take()
            inner = self.expression(_UNARY)
            return Neg(inner) if token.text == "-" else inner
        if token.kind == "op" and token.text == "(":
            self.take()
            inner = self.expression(0)
            if not self.is_op(0, ")"):
                raise self.error("missing )", self.peek())
            self.take()
            return inner
        if token.kind == "num":
            return self.number()
        if token.kind == "pct":
            self.take()
            written = token.text[:-1]
            return Num(sympy.Rational(written) / 100, token.text)
        if token.kind == "let":
            self.take()
            return Num(sympy.pi, "π") if token.text == "π" else Sym(token.text)
        if token.kind == "word":
            self.take()
            return Word(token.text.lower(), token.text)
        raise self.error(f"unexpected {token.text!r}", token)

    def number(self) -> Node:
        """A number, a typed fraction a/b, or a mixed number 3 1/2."""
        if self.mixed_at(0):
            whole, top, _, bottom = self.take(), self.take(), self.take(), self.take()
            if int(bottom.text) == 0:
                raise self.error("zero denominator", bottom)
            return Mixed(int(whole.text), int(top.text), int(bottom.text))
        token = self.take()
        num = Num(sympy.Rational(token.text), token.text)
        if not token.text.isdigit():
            return num
        # fraction: whole / whole, read as one number, unless the bottom starts a mixed number
        if self.is_op(0, "/") and self.is_whole(1) and not self.glued(2) and not self.mixed_at(1):
            self.take()
            bottom = self.take()
            return BinOp("/", num, Num(sympy.Rational(bottom.text), bottom.text), typed_fraction=True)
        return num


def unknown_runs(line: str) -> list[str]:
    """Letters written together, MIN_RUN or more, that are not a known quantity word.

    Each could be a word (rate) or letters multiplied (xyzw); only the writer knows.
    Shorter runs (3ab, 2xy) are letters multiplied, and known words (Area) and words
    in braces ({rate}) are already decided.
    """
    try:
        tokens = _tokenize(line.strip())
    except ParseError:
        return []
    runs, run = [], ""
    for token in tokens + [_Token("op", "end", len(line), True)]:
        if token.kind == "let" and token.text != "π" and run and not token.spaced:
            run += token.text
            continue
        if len(run) >= MIN_RUN and run not in runs:
            runs.append(run)
        run = token.text if token.kind == "let" and token.text != "π" else ""
    return runs


def notation_issue(line: str) -> Optional[str]:
    """Why `line` is not standard notation (docs/input_notation.md), or None if it is.

    parse() gives every line it accepts one fixed reading. This flags the lines whose
    reading rests on a convention the writer may not share (1/2x, x/2y, a^bc, x^2y,
    x^2/3, an unknown run of letters such as xyzw or fuel), and lines parse() cannot
    read at all. a/bc is standard: letters written together are one term.
    """
    try:
        parse(line)
    except ParseError as e:
        return f"cannot read: {e}"
    runs = unknown_runs(line)
    if runs:
        return f"is {runs[0]!r} one quantity or letters multiplied? write {{{runs[0]}}} for one quantity"
    tokens = _tokenize(line.strip())
    for i, token in enumerate(tokens):
        if token.kind != "op" or token.text not in ("^", "/", "÷"):
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].kind == "op" and tokens[j].text in ("-", "+"):
            j += 1  # a sign belongs to the exponent or divisor: x^-2, 6 ÷ -2
        if j >= len(tokens) or (tokens[j].kind == "op" and tokens[j].text == "("):
            continue  # a bracketed exponent or divisor is always standard
        after = tokens[j + 1] if j + 1 < len(tokens) else None
        if after is None:
            continue
        follows = after.kind in ("let", "num", "pct") or (after.kind == "op" and after.text == "(")
        if token.text == "^":
            if follows:
                return "an exponent of more than one term needs brackets: a^(bc), or a^b × c"
            if after.kind == "op" and after.text == "/":  # x^2/3 and e^2/a: is the division in the exponent?
                return "a division after an exponent needs brackets: x^(2/3) or e^(2/a), or x^2 ÷ 3"
        elif after.kind == "let" or (after.kind == "op" and after.text == "("):
            if tokens[j].kind == "let" and after.kind == "let" and not after.spaced:
                continue  # a/bc: letters written together are one term, so it means a/(bc)
            return "a divisor that mixes numbers and letters needs brackets: x/(2y), or 1/2 × x"
    return None


# ---------------------------------------------------------------- printing

_AFTER_DIVISION = re.compile(r"[/÷][\s-]*[\w.%]+$")  # text ending in a division: 22/7, a/b, 3 1/2, a/-x
_TRAILING_WHOLE = re.compile(r"(?<![\d.])\d+$")    # text ending in a whole number: x^2, x/2
_FUNCTION_LIKE = re.compile(r"[A-Za-z]{2}\s*\(")    # two letters before a bracket: read as a function call
_LETTER_RUN = re.compile(r"[A-Za-z]{2,}")


def _ends_with_power(node: Node) -> bool:
    """The printed node ends with an exponent, so nothing may be written straight after it: x^2 × y."""
    while True:
        if isinstance(node, BinOp) and node.op == "^":
            return True
        if isinstance(node, BinOp) and node.op in ("*", "/") and not as_fraction(node):
            node = node.right  # a product or division is printed ending with its right side: ÷ 7^2
        elif isinstance(node, Neg):
            node = node.operand
        else:
            return False


def _reads_as_word(text: str) -> bool:
    """Text holding letters that would be read back as a word, or flagged as a possible one."""
    return any(len(run) >= MIN_RUN or run.lower() in QUANTITY_WORDS for run in _LETTER_RUN.findall(text))


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
    if isinstance(node, Word):
        return node.text if node.name in QUANTITY_WORDS else f"{{{node.text}}}"
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
        right = right if _prec(node.right) >= p and not _fraction_like(node.right) else f"({right})"  # x^(1/2)
        return f"{left}^{right}"
    left = left if _prec(node.left) >= p else f"({left})"
    right = right if _prec(node.right) > p else f"({right})"
    if node.op == "/":
        # ÷ wherever "/" would be read back as part of a fraction (1/2 ÷ 3), or after an exponent,
        # where "/" could be read as part of it: x^2 ÷ 3 and e^2 ÷ a, not x^2/3 or e^2/a
        if (_fraction_like(node.left) or _fraction_like(node.right) or _ends_with_power(node.left)
                or (_TRAILING_WHOLE.search(left) and right[0].isdigit())):
            return f"{left} ÷ {right}"
        return f"{left}/{right}"
    if node.op == "*":
        # juxtapose only after a plain number, letter or bracket, and never after a division:
        # not 1/2x, 22/7r, 3 1/2a, a/bc (each would be read back differently) or 25%x
        joinable = (not _fraction_like(node.left) and (left[-1].isalnum() or left[-1] == ")")
                    and not _AFTER_DIVISION.search(left) and not _ends_with_power(node.left))
        # never create two letters before a bracket: xy(...) reads as a function, like sqrt(4);
        # never run letters into a word or an unknown run: a × g × e is not age, 2 × Area not 2Area
        if joinable and _prec(node.right) >= p and _show(node.right)[0].isalpha():
            joined = f"{left}{_show(node.right)}"  # 2x, 3ab, 3x^2: no brackets needed, × is associative
            if not _FUNCTION_LIKE.search(joined) and not _reads_as_word(joined):
                return joined
        if (joinable and right.startswith("(") and not _FUNCTION_LIKE.search(left + right)
                and not _reads_as_word(left + right)):
            return f"{left}{right}"  # 2(x + 3), x(y + 1)
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

    before: str    # the line before the step, as parsed
    after: str      # the student's line
    correct: str    # what the whole line should have become
    source: str     # the part of `before` the student changed
    wrong: str      # what the student wrote for that part
    expected: str   # what that part should have become
    difference: str  # wrong - expected
    wrong_term: str     # the exact term the student got wrong, as written; "(nothing)" if one is missing
    expected_term: str  # what that term should be; "(nothing)" if it should not be there
    source_node: Node = field(compare=False, repr=False)
    wrong_node: Node = field(compare=False, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        del d["source_node"], d["wrong_node"]
        return d


def locate(before: Union[str, Node], after: Union[str, Node]) -> Optional[LocatedError]:
    """Find the part of `before` the student got wrong in `after`. None if the step is correct.

    Two levels, both in the result:

    The changed part (source -> wrong), which rules run on. Parts the student copied
    or changed correctly are stripped away: terms of a sum are paired off, and the
    search goes inside brackets, products, quotients and powers while only one side
    changed. 2 × (1/2 + 1/3) -> 2 × 2/5 gives 1/2 + 1/3 -> 2/5.

    The wrong term (wrong_term -> expected_term). Both lines are expanded and
    compared monomial by monomial; each monomial that differs is pinned on the
    student's term that carries it. 2(x + 3) -> 2x + 3 gives 3, which should be 6.
    For an equation the student may have multiplied or divided both sides, so every
    factor that could explain the step is tried and the one with the fewest
    mistakes kept: 2x + 4 = 10 -> x + 4 = 5 gives 4, which should be 2, and
    x + 3 = 5 -> x = 5 + 3 gives 3, which should be -3.
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
        source, wrong = _narrow(source, wrong)
    decimal = "decimals" in _kinds(source, wrong)  # show 0.7, not 7/10, and 0.75x, not 3x/4
    if isinstance(source, Rel) or isinstance(wrong, Rel):
        expected, difference = _solved(source), ""
    else:
        value = sympy.simplify(source.to_sympy())
        if value.free_symbols:
            value = sympy.expand(value)  # 2(x + 3) should become 2x + 6
        expected = _sstr(value, decimal)
        difference = _sstr(sympy.simplify(wrong.to_sympy() - value), decimal)
    correct = _solved(b) if isinstance(b, Rel) else _sstr(sympy.simplify(b.to_sympy()), decimal)
    # a side that divides by zero (2/0) has no value, so no single term can be blamed
    term = None if undefined(source) or undefined(wrong) else _wrong_term(source, wrong, decimal)
    wrong_term, expected_term = term if term else (str(wrong), expected)
    return LocatedError(
        before=str(b), after=str(a), correct=correct,
        source=str(source), wrong=str(wrong), expected=expected, difference=difference,
        wrong_term=wrong_term, expected_term=expected_term,
        source_node=source, wrong_node=wrong,
    )


UNDEFINED = "undefined (divides by zero)"


def undefined(node: Node) -> bool:
    """True if the value divides by zero somewhere: 2/0, 6 ÷ (3 - 3)."""
    return _has_no_value(node.to_sympy())


def _has_no_value(expr) -> bool:
    return expr.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)


def _sstr(expr: sympy.Expr, decimal: bool = False) -> str:
    """SymPy's value written the way a student would (2x + 6, x^2); with decimal=True, each
    terminating fraction as a decimal (7/10 -> 0.7, 3x/4 -> 0.75x)."""
    if _has_no_value(expr):
        return UNDEFINED  # never 'zoo' or 'nan'
    if decimal and isinstance(expr, sympy.Rational) and not expr.is_integer:
        text = _as_decimal(int(expr.p), int(expr.q))
        if text is not None:
            return text
    if decimal and expr.free_symbols:  # term by term, so the decimals do not change the order of terms
        terms = [_IMPLICIT_TIMES.sub("", sympy.sstr(_decimal_symbols(t)).replace("**", "^"))
                 for t in expr.as_ordered_terms()]
        return terms[0] + "".join(f" - {t[1:]}" if t.startswith("-") else f" + {t}" for t in terms[1:])
    return _IMPLICIT_TIMES.sub("", sympy.sstr(expr).replace("**", "^"))


def _decimal_symbols(expr: sympy.Expr) -> sympy.Expr:
    """Each terminating fraction replaced by a symbol named as its decimal (3x/4 -> "0.75"·x), sign outside."""
    return expr.xreplace({r: sympy.sign(r) * sympy.Symbol(_as_decimal(abs(int(r.p)), int(r.q)))
                          for r in expr.atoms(sympy.Rational)
                          if not r.is_integer and _as_decimal(int(r.p), int(r.q)) is not None})


_IMPLICIT_TIMES = re.compile(r"(?<=[\w)])\*(?=[A-Za-z(])")  # 2*x -> 2x, 2*(x + 1) -> 2(x + 1)


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


def _narrow(b: Node, a: Node) -> tuple[Node, Node]:
    """Strip what the student kept or got right, until only the changed part is left."""
    while True:
        if _fraction_like(b) or _fraction_like(a):  # a fraction is one number: don't go inside it
            return b, a
        if isinstance(b, Neg) and isinstance(a, Neg):
            b, a = b.operand, a.operand
            continue
        if isinstance(b, BinOp) and isinstance(a, BinOp) and b.op == a.op and b.op in "*/^":
            same_left, same_right = equivalent(b.left, a.left), equivalent(b.right, a.right)
            if same_left and not same_right:
                b, a = b.right, a.right
                continue
            if same_right and not same_left:
                b, a = b.left, a.left
                continue
            return b, a
        nb, na = _unmatched_terms(b, a)
        if len(_terms(nb)) == len(_terms(b)) and len(_terms(na)) == len(_terms(a)):
            return b, a
        b, a = nb, na


# ---------------------------------------------------------------- the exact wrong term

NOTHING = "(nothing)"


@dataclass(frozen=True)
class _Term:
    side: int            # +1 left side (or a plain expression), -1 right side of a relation
    value: sympy.Expr    # its expanded value, with its sign, as it stands on its side
    text: str            # as the student wrote it, with its sign


def _side_terms(node: Node) -> list[_Term]:
    sides = [(1, node.left), (-1, node.right)] if isinstance(node, Rel) else [(1, node)]
    return [_Term(side, sympy.expand(sign * t.to_sympy()), str(t if sign > 0 else Neg(t)))
            for side, part in sides for sign, t in _terms(part)]


def _monomials(expr: sympy.Expr) -> dict[sympy.Expr, sympy.Expr]:
    """2x + 6 -> {x: 2, 1: 6}."""
    return {m: c for m, c in sympy.expand(expr).as_coefficients_dict().items() if c != 0}


def _wrong_term(source: Node, wrong: Node, decimal: bool) -> Optional[tuple[str, str]]:
    """(wrong_term, expected_term), or None if no single term can be blamed."""
    if isinstance(source, Rel) != isinstance(wrong, Rel):
        return None
    before, after = _side_terms(source), _side_terms(wrong)
    p_before = sympy.expand(sum((t.side * t.value for t in before), sympy.Integer(0)))
    p_after = sympy.expand(sum((t.side * t.value for t in after), sympy.Integer(0)))

    # Scale factors to try: 1 (terms moved or combined), and for a relation every
    # factor that makes some monomial agree (both sides multiplied or divided).
    factors = [sympy.Integer(1)]
    if isinstance(source, Rel):
        mb, ma = _monomials(p_before), _monomials(p_after)
        for m in mb.keys() & ma.keys():
            k = ma[m] / mb[m]
            if k not in factors and (source.op == "=" or k > 0):  # an inequality only keeps its direction for k > 0
                factors.append(k)
    # The student's original monomials, for judging which repair is plausible.
    seen = [(t.side, m * c) for t in before for m, c in _monomials(t.value).items()]

    best = None
    for k in factors:
        diff = _monomials(p_after - k * p_before)
        if not diff:
            continue
        pairs, unexplained = [], 0
        for m, c in diff.items():
            pick = _blame(m, m * c, after, seen, k, before)
            if pick is None:
                break
            pair, explained = pick
            pairs.append(pair)
            unexplained += not explained
        else:
            # fewest wrong terms; then mistakes in numbers over mistakes in letter terms;
            # then repairs that match something the student started with; then no scaling
            score = (len(diff), sum(m != 1 for m in diff), unexplained, k != 1)
            if best is None or score < best[0]:
                best = (score, pairs)
    if best is None:
        return None
    pairs = best[1]
    extra = [p for p in pairs if p[1] == NOTHING]
    missing = [p for p in pairs if p[0] == NOTHING]
    if len(extra) == 1 and len(missing) == 1:  # 2 should not be there and 2x is missing: 2 should be 2x
        pairs = [p for p in pairs if p not in (extra[0], missing[0])] + [(extra[0][0], missing[0][1])]
    return ", ".join(_show_value(w, decimal) for w, _ in pairs), ", ".join(_show_value(e, decimal) for _, e in pairs)


def _blame(monomial, delta, after: list[_Term], seen, k, before: list[_Term]):
    """Which of the student's terms carries `delta` (the extra amount on monomial), and its repair.

    Returns ((wrong, expected), explained) or None if a term that mixes several
    monomials (an unexpanded bracket) is in the way.
    """
    holders = []
    for t in after:
        mono = _monomials(t.value)
        if monomial in mono:
            if len(mono) > 1:
                return None
            holders.append(t)
    if not holders:
        return (NOTHING, -delta), False  # a term is missing
    copied = {(t.side, t.text) for t in before}

    def plausible(t: _Term) -> bool:
        repaired = t.value - t.side * delta
        return any(sympy.simplify(repaired - k * value * side * t.side) == 0 for side, value in seen)

    holders.sort(key=lambda t: (not plausible(t), (t.side, t.text) in copied if k == 1 else 0))
    t = holders[0]
    repaired = sympy.expand(t.value - t.side * delta)
    return (t.text, NOTHING if repaired == 0 else repaired), plausible(t)


def _show_value(v, decimal: bool) -> str:
    return v if isinstance(v, str) else _sstr(v, decimal)


def _kinds(*nodes: Node) -> set[str]:
    """Which of 'expressions' (a letter) and 'decimals' (a decimal number or %) appear in the nodes."""
    found = set()

    def walk(n: Node):
        if isinstance(n, (Sym, Word)):
            found.add("expressions")
        elif isinstance(n, Num) and ("." in n.text or "%" in n.text):  # percentages are in the decimals topic
            found.add("decimals")
        for child in (getattr(n, k, None) for k in ("operand", "left", "right")):
            if child is not None:
                walk(child)

    for n in nodes:
        walk(n)
    return found


def topic_of(*nodes: Node) -> str:
    """expressions if any letter appears, decimals if any decimal number does, else fractions."""
    found = _kinds(*nodes)
    return "expressions" if "expressions" in found else "decimals" if "decimals" in found else "fractions"
