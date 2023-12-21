# data

**load method**:
```python
import json
data = json.load(open(path_to_file, 'r'))
print(data.keys())
```

+ `outputs`:  Our input safety-sentitive documents (llm-attack outputs).
+ `models`: llm-attack --> backbone model.
+ `goals`: llm-attack --> original harmful prompt.
+ `questions`: close-domain question generated by [QG-model](https://github.com/asahi417/lm-question-generation) (used in close-domain QA task).