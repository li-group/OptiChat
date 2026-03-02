from google.adk.models.lite_llm import LiteLlm
from optichat.config.llm_cfg import *

gpt_5 = LiteLlm(model=GPT_5,
                temperature=GPT_5_TEMPERATURE,
                max_tokens=GPT_5_MAX_TOKENS,
                reasoning_effort="minimal",
                verbosity="low")
gpt_5_mini = LiteLlm(model=GPT_5_MINI,
                     temperature=GPT_5_MINI_TEMPERATURE,
                     max_tokens=GPT_5_MINI_MAX_TOKENS,
                     reasoning_effort="minimal",
                     verbosity="medium")
gpt_5_nano = LiteLlm(model=GPT_5_NANO,
                     temperature=GPT_5_NANO_TEMPERATURE,
                     max_tokens=GPT_5_NANO_MAX_TOKENS)
gpt_4o = LiteLlm(model=GPT_4o,
                 temperature=GPT_4o_TEMPERATURE,
                 max_tokens=GPT_4o_MAX_TOKENS)
gpt_4o_mini = LiteLlm(model=GPT_4o_MINI,
                     temperature=GPT_4o_MINI_TEMPERATURE,
                     max_tokens=GPT_4o_MINI_MAX_TOKENS)
gpt_5_codex = LiteLlm(model=GPT_5_CODEX,
                      temperature=GPT_5_CODEX_TEMPERATURE,
                      max_tokens=GPT_5_CODEX_MAX_TOKENS,
                      reasoning_effort="low")
gpt_5_1 = LiteLlm(model=GPT_5_1,
                      temperature=GPT_5_1_TEMPERATURE,
                      max_tokens=GPT_5_1_MAX_TOKENS,
                      reasoning_effort="none",
                      verbosity="low")
gpt_5_2_codex = LiteLlm(model=GPT_5_2_CODEX,
                      temperature=GPT_5_2_CODEX_TEMPERATURE,
                      max_tokens=GPT_5_2_CODEX_MAX_TOKENS)                      
