from dataclasses import dataclass, field

@dataclass
class DetectedTable:
    type_table: str
    table_name: str
    start_line: int
    end_line: int
    lines: list[str]
    # Nearby heading/unit text is available for metadata detection but is never parsed as rows.
    context_lines: list[str] = field(default_factory=list)
