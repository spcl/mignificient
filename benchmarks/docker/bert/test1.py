import json
from bert import QA
from timeit import default_timer as timer

before = timer()
model = QA('/app/model')
after = timer()

print('model eval time:')
print(after - before)

start = timer()

doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
      "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
      "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
      "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
      "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
      "people in a referendum, depending on the provision. "

q = 'When did Victoria enact its constitution?'

# Store doc and question into a JSON 
json_obj = {
      "passage": "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
            "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
            "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
            "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
            "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
            "people in a referendum, depending on the provision. ",
      "question": 'When did Victoria enact its constitution?',
}

payload = json.dumps(json_obj)

answer = model.predict(payload)
print(answer['answer'])
print(answer.keys())

end = timer()
print(end - start)
