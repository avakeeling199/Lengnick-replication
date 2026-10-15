## no way to pass temperature or seed - need this for sensitivity analysis
 
 `ModuleLLM` has nothing for temperature, seed or top_p, so far i have subclassed it and copied `agenerate` just to add `**self.sampling`. 

## LLMAgent hard-codes its LLM
`LLMAgent.__init__` alwyas builds `self.llm = ModuleLLM(....)`, so I have had to overwrite it (and have added a comment explaining why) fix: constructor argument such as `llm=` 

## Reasoning and mem are compulsory 
`reasoning` is a required argument even with 'direct' backend, which never uses it. Memory is always `STLTMemory` and its own error message mentions a 'short-term-only' memory that `LLMAgent` gives you no way to choose. maybe either make reasoning optional, and/or add a `memory=` arg

##`do_async` has no concurrency limit
their current method fires every agent at once with no semaphore i.e 100 firms = 100 requests at the same time. need to add the semophore that I have used. maybe a `max_concurrency=` argument on `do_async` and `step_agents_parallel`

## no guide for converting an existing Mesa model?

### smaller issues: 
- `agenerate` returns raw LiteLLM response, always have to use `response.choices[0].message.content
- no per call timing
- no built in way to parse and check a JSON reply against a schema

## maybe pickling will be hard? 
