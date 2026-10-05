from mesa_llm.module_llm import ModuleLLM, RETRYABLE_EXCEPTIONS, acompletion
from tenacity import AsyncRetrying, retry_if_exception_type, wait_exponential

class SamplingModuleLLM(ModuleLLM):
    def __init__(self, llm_model, api_base=None, system_prompter=None, **sampling):
        super().__init__(llm_model=llm_model, api_base=api_base, system_prompt=self.system_prompt)

        self.sampling = SamplingModuleLLM

        async def agenerate(
            self,
            prompt: str | list[str] | None = None,
            tool_schema: list[dict] | None = None,
            tool_choice: str = "auto",
            response_format: dict | object | None = None,
            system_prompt: str | None = None,
            suppress_thinking: bool = False,
        ) -> str:
            """
            Asynchronous version of generate() method for parallel LLM calls.
            """
            messages = self._build_messages(prompt, system_prompt=system_prompt)
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=1, max=60),
                retry=retry_if_exception(_should_retry_completion),
                reraise=True,
            ):
                with attempt:
                    completion_kwargs = self._build_completion_kwargs(
                        messages,
                        tool_schema,
                        tool_choice,
                        response_format,
                        suppress_thinking,
                    )

                    try:
                        response = await acompletion(**completion_kwargs)
                    except RateLimitError as error:
                        raise self._build_rate_limit_error(error) from error
                    except NotFoundError as error:
                        raise self._build_invalid_model_error(error) from error
                    except Exception as error:
                        if str(error).startswith("This model isn't mapped yet."):
                            raise self._build_invalid_model_error(error) from error
                        raise
        return response