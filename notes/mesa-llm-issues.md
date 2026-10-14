## no way to pass temperature or seed - need this for sensitivity analysis
 
 ModuleLLM has nothing for temperature, seed or top_p, so far i have subclassed it and copied agenerate just to add **self.sampling. 

## LLMAgent hard-codes its LLM

