from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    bot_token: str = Field(validation_alias="BOT_TOKEN")
    database_url: str = Field(validation_alias="DATABASE_URL")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    agentplatform_key: str | None = Field(
        default=None, validation_alias="AGENTPLATFORM_KEY"
    )


settings = Settings()
