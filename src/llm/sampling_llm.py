from mesa_llm.module_llm import ModuleLLM, RETRYABLE_EXCEPTIONS, acompletion
from tenacity import AsyncRetrying, retry_if_exception_type, wait_exponential

class SamplingModuleLLM(ModuleLLM):
        # note that this is different to how it would have to be to PR with mesa-llm because this is an earlier 
        # released version, what is live on github is different
    def __init__(self, llm_model, api_base=None, system_prompt=None, **sampling):
        super().__init__(llm_model=llm_model, api_base=api_base, system_prompt=system_prompt)

        self.sampling = sampling

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                completion_kwargs = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": tool_schema,
                    "tool_choice": tool_choice if tool_schema else None,
                    "response_format": response_format,
                    **self.sampling,
                }
                if self.api_base:
                    completion_kwargs["api_base"] = self.api_base
                response = await acompletion(**completion_kwargs)

        return response