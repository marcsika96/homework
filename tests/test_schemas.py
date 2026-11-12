import pytest
from pydantic import ValidationError

from reviewgraph.schemas import (
    Finding,
    ReviewOutput,
    ReviewResult,
    Verification,
    clean_findings,
)

PATH = "app/studio/usecases/booking.py"
TEN_LINES = "\n".join(f"line {number}" for number in range(1, 11)) + "\n"


def make_finding(line_start: int = 3, line_end: int = 4, file: str = PATH) -> Finding:
    return Finding(
        file=file,
        line_start=line_start,
        line_end=line_end,
        category="authorization",
        severity="high",
        explanation="Any signed-in user can cancel the booking of another dancer.",
        fix="Check that the booking belongs to the current user before cancelling it.",
    )


def test_correct_finding_is_left_unchanged():
    finding = make_finding(3, 4)

    assert clean_findings([finding], TEN_LINES, PATH) == [finding]


def test_reversed_line_range_is_swapped():
    cleaned = clean_findings([make_finding(7, 2)], TEN_LINES, PATH)[0]

    assert (cleaned.line_start, cleaned.line_end) == (2, 7)


@pytest.mark.parametrize(
    "given, expected",
    [
        ((8, 250), (8, 10)),     # end after the last line
        ((40, 50), (10, 10)),    # whole range after the last line
        ((0, 3), (1, 3)),        # models sometimes count from 0
        ((-5, -1), (1, 1)),
    ],
)
def test_lines_are_moved_into_the_file(given, expected):
    cleaned = clean_findings([make_finding(*given)], TEN_LINES, PATH)[0]

    assert (cleaned.line_start, cleaned.line_end) == expected


def test_file_is_set_to_the_reviewed_path():
    cleaned = clean_findings([make_finding(file="booking.py")], TEN_LINES, PATH)[0]

    assert cleaned.file == PATH


def test_empty_file_gives_line_one():
    cleaned = clean_findings([make_finding(5, 9)], "", PATH)[0]

    assert (cleaned.line_start, cleaned.line_end) == (1, 1)


def test_original_findings_are_not_modified():
    finding = make_finding(7, 2, file="booking.py")

    clean_findings([finding], TEN_LINES, PATH)

    assert (finding.file, finding.line_start, finding.line_end) == ("booking.py", 7, 2)


def test_no_findings_stay_no_findings():
    assert clean_findings([], TEN_LINES, PATH) == []


def test_unknown_category_and_severity_are_rejected():
    values = make_finding().model_dump()

    with pytest.raises(ValidationError):
        Finding(**{**values, "category": "style"})
    with pytest.raises(ValidationError):
        Finding(**{**values, "severity": "critical"})


@pytest.mark.parametrize("schema", [Finding, ReviewOutput, Verification])
def test_every_model_facing_field_has_a_description(schema):
    for name, field in schema.model_fields.items():
        assert field.description, f"{schema.__name__}.{name} has no description"


def test_review_result_starts_empty_and_survives_json():
    result = ReviewResult(unit_id=f"I1:{PATH}", system="baseline", model="gemini-3.8-flash")
    other = ReviewResult(unit_id=f"I2:{PATH}", system="agent", model="gemini-3.8-flash")

    # The default lists and stats must not be shared between results.
    result.findings.append(make_finding())
    result.stats.llm_calls += 1
    assert other.findings == [] and other.stats.llm_calls == 0

    assert ReviewResult.model_validate_json(result.model_dump_json()) == result
