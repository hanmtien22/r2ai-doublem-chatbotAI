from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RetrievalQuery:
    ticker: str
    year: int
    section: str
    indicator_code: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FormulaInfo:
    name: str
    formula: str
    components: list[str]
    unit: str
    name_en: Optional[str] = None
    multiply_100: bool = False
    requires_previous_year: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedEntities:
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    indicator_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataFilters:
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    original_question: str
    normalized_question: str
    entities: ExtractedEntities
    query_type: str
    requires_formula: bool
    formula_info: Optional[FormulaInfo]
    retrieval_queries: list[RetrievalQuery]
    search_text: str
    metadata_filters: MetadataFilters

    def to_dict(self) -> dict:
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "entities": self.entities.to_dict(),
            "query_type": self.query_type,
            "requires_formula": self.requires_formula,
            "formula_info": self.formula_info.to_dict() if self.formula_info else None,
            "retrieval_queries": [q.to_dict() for q in self.retrieval_queries],
            "search_text": self.search_text,
            "metadata_filters": self.metadata_filters.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
