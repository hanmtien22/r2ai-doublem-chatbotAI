from dataclasses import dataclass

@dataclass
class DetectedTable:
    type_table: str
    table_name: str
    start_line: int
    end_line: int
    lines: list[str]