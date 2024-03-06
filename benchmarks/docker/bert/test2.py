from bert import QA
from timeit import default_timer as timer

before = timer()
model = QA('/app/model')
after = timer()

print('model eval time:')
print(after - before)

start = timer()

doc = 'At the 52nd Annual Grammy Awards, Beyoncé received ten nominations, including Album of the Year for I Am... ' \
      'Sasha Fierce, Record of the Year for "Halo", and Song of the Year for "Single Ladies (Put a Ring on It)", ' \
      'among others. She tied with Lauryn Hill for most Grammy nominations in a single year by a female artist. In ' \
      '2010, Beyoncé was featured on Lady Gaga\'s single "Telephone" and its music video. The song topped the US Pop ' \
      'Songs chart, becoming the sixth number-one for both Beyoncé and Gaga, tying them with Mariah Carey for most ' \
      'number-ones since the Nielsen Top 40 airplay chart launched in 1992. "Telephone" received a Grammy Award ' \
      'nomination for Best Pop Collaboration with Vocals. '

q = 'How many awards was Beyonce nominated for at the 52nd Grammy Awards?'

answer = model.predict(doc,q)

# print(answer['answer'])
# print(answer.keys())

end = timer()
print(end - start)
