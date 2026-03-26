from pydantic import BaseModel, model_validator, Field

class AnalysisResult(BaseModel):
    page_type: str | None = Field(default=None, validation_alias="pagetype")
    important_points: list[str | dict] = Field(default=[], validation_alias="importantpoints")
    summary: str | None = None
    key_facts: list[str | dict] = Field(default=[], validation_alias="keyfacts")
    pricing_info: list[str | dict] = Field(default=[], validation_alias="pricinginfo")
    feature_comparison: list[str | dict] = Field(default=[], validation_alias="featurecomparison")
    tables_detected: list[str | dict] = Field(default=[], validation_alias="tablesdetected")
    missing_structured_data: list[str | dict] = Field(default=[], validation_alias="missingstructureddata")
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def match_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        aliases = {
            "pagetype": "page_type",
            "importantpoints": "important_points",
            "keyfacts": "key_facts",
            "pricinginfo": "pricing_info",
            "featurecomparison": "feature_comparison",
            "tablesdetected": "tables_detected",
            "missingstructureddata": "missing_structured_data"
        }
        for wrong, right in aliases.items():
            if wrong in data and right not in data:
                data[right] = data.pop(wrong)
        return dict(data)

    @model_validator(mode="after")
    def flatten_dicts(self):
        def _flatten(items):
            res = []
            for i in items:
                if isinstance(i, dict):
                    # If it's a dict, join its values as a string, e.g. {"fact": "hello"} -> "hello"
                    res.append(" - ".join(str(v) for v in i.values() if v))
                else:
                    res.append(str(i))
            return res

        self.important_points = _flatten(self.important_points)
        self.key_facts = _flatten(self.key_facts)
        self.missing_structured_data = _flatten(self.missing_structured_data)
        
        return self


class NewsItem(BaseModel):
    id: int | None = None
    title: str | None = None
    url: str | None = None
    content: str | None = None
    raw_content: str | None = None
    analysis: AnalysisResult | None = None


class AgentRequest(BaseModel):
    topic: str | None = None
    url: str | None = None
    message: str | None = None
    thread_id: str | None = None

    @model_validator(mode="after")
    def validate_input(self):
        def clean(value: str | None) -> str | None:
            v = (value or "").strip()
            return None if v == "" or v == "string" else v

        self.topic = clean(self.topic)
        self.url = clean(self.url)
        self.message = clean(self.message)
        self.thread_id = clean(self.thread_id)

        if not self.topic and not self.url and not self.message:
            raise ValueError("Debes enviar topic, url o message")

        return self


class AgentResponse(BaseModel):
    thread_id: str
    topic: str | None = None
    results: list[NewsItem]
    error: str | None = None
    assistant_message: str | None = None