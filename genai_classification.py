from transformers import pipeline

candidate_labels = ["food", "tech", "politics", "business", "finance", "art", "geography", "sports"]

my_classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')


english_texts = [
     "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
     "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
     "Canada has the longest coastline of any country in the world, stretching over 202,080 kilometers (125,567 miles).",
     "The longest novel ever written is 'In Search of Lost Time' by Marcel Proust. It has approximately 1.2 million words.",
     "The first 1GB hard drive, released in 1980 by IBM, weighed over 500 pounds and cost $40,000.",
     "The world’s largest grand piano was built by a 15-year-old in New Zealand. It is 5.7 meters long and took four years to build.",
     "Leonardo da Vinci's 'Mona Lisa' has no eyebrows because it was the fashion in Renaissance Florence to shave them off.",
     "A single strand of spider silk is five times stronger than a strand of steel of the same thickness.",
     "The longest tennis match in history took place at Wimbledon in 2010 between John Isner and Nicolas Mahut, lasting 11 hours and 5 minutes over three days.",
     "There is a giant cloud of alcohol in Sagittarius B, a gas cloud in the Milky Way, containing enough ethyl alcohol to make 400 trillion trillion pints of beer.",
]

text_length = len(english_texts)
for text_index in range(text_length):
  prediction = my_classifier(english_texts[text_index], candidate_labels, multi_label = True)
  print('text',text_index,' : ',prediction['sequence'])
  labels_length = len(candidate_labels)
  for label_item in range(labels_length):
     print('{:>10} : {:>10}'.format(prediction['labels'][label_item],prediction['scores'][label_item]))
  print("")
