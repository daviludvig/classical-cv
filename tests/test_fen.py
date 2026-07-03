"""Test suite for FEN generation and helpers (src.chess).

Runnable without pytest:  python tests/test_fen.py
Also works under pytest:   pytest tests/test_fen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess import (  # noqa: E402
    piece_to_fen_symbol, board_placement_fen, board_to_fen,
    castling_rights_from_position, _normalize_castling, _validate_en_passant,
)

_BACK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]


def start_map():
    """Standard starting position as a {square: label} map."""
    m = {}
    for i, f in enumerate("ABCDEFGH"):
        m[f"{f}8"] = _BACK[i] + "_b"
        m[f"{f}7"] = "pawn_b"
        m[f"{f}2"] = "pawn_w"
        m[f"{f}1"] = _BACK[i] + "_w"
    return m


def moved(m, **changes):
    """Copy map, applying {square: label|None} changes (None removes)."""
    out = dict(m)
    for sq, label in changes.items():
        if label is None:
            out.pop(sq, None)
        else:
            out[sq] = label
    return out


# --- test registry ----------------------------------------------------------
_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # wrong exception type
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# --- piece symbols ----------------------------------------------------------
@test
def piece_symbols_all_twelve():
    expect = {
        "pawn_w": "P", "knight_w": "N", "bishop_w": "B",
        "rook_w": "R", "queen_w": "Q", "king_w": "K",
        "pawn_b": "p", "knight_b": "n", "bishop_b": "b",
        "rook_b": "r", "queen_b": "q", "king_b": "k",
    }
    for label, sym in expect.items():
        assert piece_to_fen_symbol(label) == sym, label


# --- placement --------------------------------------------------------------
@test
def placement_start():
    assert board_placement_fen(start_map()) == \
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@test
def placement_empty_board():
    assert board_placement_fen({}) == "8/8/8/8/8/8/8/8"


@test
def placement_single_piece_corner():
    assert board_placement_fen({"A1": "rook_w"}) == "8/8/8/8/8/8/8/R7"
    assert board_placement_fen({"H8": "king_b"}) == "7k/8/8/8/8/8/8/8"


@test
def placement_mixed_rank_counts_empties():
    # rank 8: rook a8, empty, bishop c8, rest empty
    m = {"A8": "rook_b", "C8": "bishop_w"}
    assert board_placement_fen(m).split("/")[0] == "r1B5"


@test
def placement_after_e4():
    m = moved(start_map(), E2=None, E4="pawn_w")
    assert board_placement_fen(m) == \
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


@test
def placement_after_e4_c5():
    m = moved(start_map(), E2=None, E4="pawn_w", C7=None, C5="pawn_b")
    assert board_placement_fen(m) == \
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR"


# --- full FEN default (backward compatible) ---------------------------------
@test
def full_fen_defaults():
    assert board_to_fen(start_map()) == \
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


@test
def full_fen_start_with_all_fields():
    assert board_to_fen(start_map(), side="w", castling="KQkq") == \
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@test
def full_fen_after_e4_known():
    m = moved(start_map(), E2=None, E4="pawn_w")
    assert board_to_fen(m, side="b", castling="KQkq", en_passant="e3") == \
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


@test
def full_fen_after_e4_c5_known():
    m = moved(start_map(), E2=None, E4="pawn_w", C7=None, C5="pawn_b")
    assert board_to_fen(m, side="w", castling="KQkq", en_passant="c6",
                        halfmove=0, fullmove=2) == \
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"


# --- side --------------------------------------------------------------------
@test
def side_black():
    assert board_to_fen({}, side="b").split()[1] == "b"


@test
def side_invalid_raises():
    raises(ValueError, lambda: board_to_fen({}, side="x"))
    raises(ValueError, lambda: board_to_fen({}, side="white"))


# --- castling ----------------------------------------------------------------
@test
def castling_canonical_order():
    assert _normalize_castling("qkQK") == "KQkq"
    assert _normalize_castling("kq") == "kq"
    assert _normalize_castling("Kq") == "Kq"


@test
def castling_none_forms():
    assert _normalize_castling("-") == "-"
    assert _normalize_castling("") == "-"


@test
def castling_invalid_char_raises():
    raises(ValueError, lambda: _normalize_castling("X"))
    raises(ValueError, lambda: _normalize_castling("KQx"))


@test
def castling_duplicate_raises():
    raises(ValueError, lambda: _normalize_castling("KK"))
    raises(ValueError, lambda: _normalize_castling("KQkqK"))


@test
def castling_field_in_full_fen():
    assert board_to_fen({}, castling="qK").split()[2] == "Kq"


# --- en passant --------------------------------------------------------------
@test
def en_passant_valid():
    assert _validate_en_passant("e3") == "e3"
    assert _validate_en_passant("E6") == "e6"  # lowercased
    assert _validate_en_passant("-") == "-"
    assert _validate_en_passant("a3") == "a3"
    assert _validate_en_passant("h6") == "h6"


@test
def en_passant_invalid_raises():
    raises(ValueError, lambda: _validate_en_passant("e4"))  # wrong rank
    raises(ValueError, lambda: _validate_en_passant("e5"))
    raises(ValueError, lambda: _validate_en_passant("z3"))  # bad file
    raises(ValueError, lambda: _validate_en_passant("e"))   # too short
    raises(ValueError, lambda: _validate_en_passant("e33"))  # too long


# --- clocks ------------------------------------------------------------------
@test
def clocks_custom_values():
    assert board_to_fen({}, halfmove=12, fullmove=34).split()[-2:] == ["12", "34"]


@test
def clocks_accept_str_ints():
    assert board_to_fen({}, halfmove="5", fullmove="7").endswith(" 5 7")


@test
def clocks_invalid_raise():
    raises(ValueError, lambda: board_to_fen({}, halfmove=-1))
    raises(ValueError, lambda: board_to_fen({}, fullmove=0))
    raises(ValueError, lambda: board_to_fen({}, fullmove=-3))


# --- castling from position heuristic ---------------------------------------
@test
def castling_from_start_position():
    assert castling_rights_from_position(start_map()) == "KQkq"


@test
def castling_from_position_king_moved():
    m = moved(start_map(), E1=None, F1="king_w")  # white king off e1
    assert castling_rights_from_position(m) == "kq"


@test
def castling_from_position_rook_missing():
    m = moved(start_map(), H1=None)  # white kingside rook gone
    assert castling_rights_from_position(m) == "Qkq"


@test
def castling_from_position_empty():
    assert castling_rights_from_position({}) == "-"


@test
def castling_from_position_feeds_board_to_fen():
    m = start_map()
    fen = board_to_fen(m, side="w", castling=castling_rights_from_position(m))
    assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- runner ------------------------------------------------------------------
def run():
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{passed + failed} tests passed"
          + ("" if not failed else f"  ({failed} FAILED)"))
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
