# System Overview


![System Flow Chart](..\assets\system_design.svg)

### Implementation Notes:

1. Running the LLMs as separate services would enable better scaling (i.e. using `OllamaModel` or `OpenAIModel`)
2. Each models handles their own prompt construction
3. The ensemble step is only triggered when using multiple models 
4. Currently not passing the translation memories directly to the ensemble model


