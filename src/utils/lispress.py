
def try_round_trip(lispress_str: str) -> str:
    """
    If `lispress_str` is valid lispress, round-trips it to and from `Program`.
    This puts named arguments in alphabetical order and normalizes numbers
    so they all have exactly one decimal place.
    If it is not valid, returns the original string unmodified.
    """
    try:
        return _try_round_trip(lispress_str)
    except Exception:  # pylint: disable=W0703
        return lispress_str


def _try_round_trip(lispress_str: str) -> str:
    # round-trip to canonicalize
    lispress = parse_lispress(lispress_str)
    program, _ = lispress_to_program(lispress, 0)
    round_tripped = program_to_lispress(program)

    def normalize_numbers(exp: Lispress) -> "Lispress":
        if isinstance(exp, str):
            try:
                num = float(exp)
                return f"{num:.1f}"
            except ValueError:
                return exp
        else:
            return [normalize_numbers(e) for e in exp]

    return render_compact(strip_copy_strings(normalize_numbers(round_tripped)))