"""Compare reduced-dimensionality methods: LSA vs NMF vs autoencoder.

Each pipeline ends with the same LogisticRegression head — only the
bottleneck featurizer differs.
"""

# %%
from jurebes.benchmark import compare

greet = ["hello world", "hi there", "hey friend", "good morning sunshine",
         "howdy partner", "salutations", "morning everyone", "greetings folks",
         "good day", "hello again friend", "hey buddy now", "morning star bright",
         "afternoon hello there", "evening greetings folks", "good day sir", "hi mate today",
         "morning sunshine bright", "hello dear friend", "hey there pal", "hi everyone here"]
bye = ["goodbye now", "see you later", "farewell friend dear", "catch you soon now",
       "take care friend", "until next time", "so long buddy", "ciao for now",
       "later alligator friend", "adieu my dear", "bye now folks", "talk soon friend",
       "in a while", "goodbye forever", "off i go", "see ya soon",
       "bye bye now", "farewell dear", "until we meet", "stay safe out"]
thanks = ["thank you", "thanks a lot", "much appreciated indeed", "cheers mate",
          "grateful for help", "kind of you", "many thanks friend", "ta indeed",
          "appreciate it greatly", "obliged sir", "thanks so much", "thank you kindly",
          "warm thanks here", "you are kind", "great help indeed", "appreciate the help",
          "kindly appreciated", "very grateful", "thanks again friend", "with gratitude all"]
weather = ["how is the weather", "is it sunny outside", "raining today outside",
           "weather forecast please", "current temperature now", "humidity level today",
           "wind speed report", "warm or cold outside", "snowing now", "cloudy skies above",
           "foggy this morning", "is it nice outside", "hot summer day", "cool autumn breeze",
           "winter chill outside", "spring warmth here", "is it freezing", "muggy weather",
           "windy day outside", "drizzle this morning"]

X = greet + bye + thanks + weather
y = (["greet"] * len(greet) + ["bye"] * len(bye)
     + ["thanks"] * len(thanks) + ["weather"] * len(weather))

names = ["lsa_logreg", "nmf_logreg", "lda_logreg", "autoencoder_logreg"]
result = compare(names, X, y, k=3)
for r in result.rows:
    print(f"{r.name:22s} macro_f1={r.macro_f1:.3f} train={r.train_seconds:.2f}s")
