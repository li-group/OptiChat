from google.adk.models.lite_llm import LiteLlm
from optichat.config.llm_cfg import *

gpt_5 = LiteLlm(model=GPT_5,
                temperature=GPT_5_TEMPERATURE,
                max_tokens=GPT_5_MAX_TOKENS)
gpt_5_mini = LiteLlm(model=GPT_5_MINI,
                     temperature=GPT_5_MINI_TEMPERATURE,
                     max_tokens=GPT_5_MINI_MAX_TOKENS)
gpt_5_nano = LiteLlm(model=GPT_5_NANO,
                     temperature=GPT_5_NANO_TEMPERATURE,
                     max_tokens=GPT_5_NANO_MAX_TOKENS)

