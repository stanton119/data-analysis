# Pydantic exploration


```python
from datetime import datetime

from pydantic import BaseModel, PositiveInt


class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]


external_data = {
    "id": 123,
    "signup_ts": "2019-06-01 12:22",
    "tastes": {
        "wine": 9,
        b"cheese": 7,
        "cabbage": "1",
    },
}

user = User(**external_data)

print(user.id)
# > 123
print(user.model_dump())
"""
{
    'id': 123,
    'name': 'John Doe',
    'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
    'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1},b
}
"""
```

    123
    {'id': 123, 'name': 'John Doe', 'signup_ts': datetime.datetime(2019, 6, 1, 12, 22), 'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1}}





    "\n{\n    'id': 123,\n    'name': 'John Doe',\n    'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),\n    'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1},b\n}\n"



Creating objects with the wrong types


```python
external_data = {
    "id": "a",
    "name": 42,
    "signup_ts": "2019-06-01 12:22",
    "tastes": {
        "wine": 9,
        b"cheese": 7,
        "cabbage": "1",
    },
}
external_data = {
    "id": "a",
    "name": 42,
    "signup_ts": "2019-06-01 12:22",
}

user = User(**external_data)
user
```


    ---------------------------------------------------------------------------

    ValidationError                           Traceback (most recent call last)

    Cell In[2], line 17
          1 external_data = {
          2     'id': "a",
          3     "name": 42,
       (...)      9     },
         10 }
         11 external_data = {
         12     'id': "a",
         13     "name": 42,
         14     'signup_ts': '2019-06-01 12:22',  
         15 }
    ---> 17 user = User(**external_data)  
         18 user


    File ~/Developer/Github/VariousDataAnalysis/tools_python/pydantic_exploration/.venv/lib/python3.12/site-packages/pydantic/main.py:250, in BaseModel.__init__(self, **data)
        248 # `__tracebackhide__` tells pytest and some other tools to omit this function from tracebacks
        249 __tracebackhide__ = True
    --> 250 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        251 if self is not validated_self:
        252     warnings.warn(
        253         'A custom validator is returning a value other than `self`.\n'
        254         "Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\n"
        255         'See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
        256         stacklevel=2,
        257     )


    ValidationError: 3 validation errors for User
    id
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
        For further information visit https://errors.pydantic.dev/2.12/v/int_parsing
    name
      Input should be a valid string [type=string_type, input_value=42, input_type=int]
        For further information visit https://errors.pydantic.dev/2.12/v/string_type
    tastes
      Field required [type=missing, input_value={'id': 'a', 'name': 42, '...ts': '2019-06-01 12:22'}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.12/v/missing

