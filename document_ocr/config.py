from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")

    # OpenTyphoon cloud API (free for research — https://opentyphoon.ai)
    typhoon_api_key: str = Field(default="")
    typhoon_api_base_url: str = Field(default="https://api.opentyphoon.ai/v1")
    typhoon_api_model: str = Field(default="typhoon-ocr")

    # Ollama self-hosted
    ollama_base_url: str = Field(default="http://localhost:11434")
    typhoon_ocr_model: str = Field(default="scb10x/typhoon-ocr1.5-3b")
    gemma_struct_model: str = Field(default="gemma3:4b")

    # Preprocessing
    pdf_dpi: int = Field(default=300)
    vlm_max_side_px: int = Field(default=1800)
    deskew_enabled: bool = Field(default=True)

    # Evaluation
    cpu_cost_per_hour_usd: float = Field(default=0.05)
    llm_judge_enabled: bool = Field(default=True)


settings = Settings()
