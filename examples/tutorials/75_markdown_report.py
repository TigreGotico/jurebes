"""Render a comparison as a markdown table.

to_markdown supports sort_by and precision arguments. The group column
is rendered alongside each baseline name.
"""

# %%
from jurebes.benchmark import compare, to_markdown

X = ["hello", "hi", "bye", "goodbye", "thanks", "thank you"] * 5
y = (["greet"] * 2 + ["bye"] * 2 + ["thanks"] * 2) * 5

result = compare(["logreg", "linear_svc", "nb_multinomial"], X, y, k=2)
md = to_markdown(result, sort_by="macro_f1", precision=3)
print(md)
