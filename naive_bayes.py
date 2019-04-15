
import tarfile 
import re
import operator
from nltk.corpus import stopwords 
from collections import defaultdict 

print('\n') 

sw = set(stopwords.words('english'))
sw.update('\n')

         
#read in training files:    
         
train = []
test = []
gold = []
dictn = []
tar = tarfile.open("/Users/aneeshnaik/Downloads/train.tar.gz","r:gz")
tar2 = tarfile.open("/Users/aneeshnaik/Downloads/test.tar.gz","r:gz")
tar3 = tarfile.open("/Users/aneeshnaik/Downloads/gold.tar.gz","r:gz")
tar4 = tarfile.open("/Users/aneeshnaik/Downloads/dict.tar.gz","r:gz")
#i=1

#creating word map:

words = []
for name in tar.getnames():
  words.append(name[6:-4])
  
test_words = []
for name in tar2.getnames():
  test_words.append(name[5:-5])
 
del[words[0]]

del[test_words[0]]
 
word_map = []

for i in range(0,len(words)):
  word_map.append((words[i],i+1))
  
for member in tar.getmembers():
     f = tar.extractfile(member)
     if f is not None:
         train.append(f)
         
for member in tar2.getmembers():
     f = tar2.extractfile(member)
     if f is not None:
         test.append(f)
         
for member in tar3.getmembers():
     f = tar3.extractfile(member)
     if f is not None:
         gold.append(f)

for member in tar4.getmembers():
     f = tar4.extractfile(member)
     if f is not None:
         dictn.append(f)
                
###############################################################################
 
# getting user input:
           
print("Choose a word from the list below (enter it's corresponding number):",'\n')
i = 1
for item in word_map:
  print(i, item[0])
  i = i+1
  
entry = input("Enter choice: ")
selected_word = word_map[int(entry)-1][0]

for i in range(0,len(tar4.getnames())):
  if(selected_word==tar4.getnames()[i][5:-4]):
    dict_pos = i
  


# choosing appropriate training and test data:

pos = []

for i in range(0, len(test_words)):
  if(selected_word == test_words[i][0:-2]):
    pos.append(i)
    
#for item in dictn:
  #print(item.read().decode("utf-8"),'\n')

selected_training = train[int(entry)-1].read().decode("utf-8")
selected_training = re.sub('8\d\d\d\d\d','',selected_training)
selected_dict = dictn[int(dict_pos)-1].read().decode("utf-8")


if(len(pos)>1):
  t1 = test[int(pos[0])].read().decode("utf-8")
  t2 = test[int(pos[1])].read().decode("utf-8")
  g1 = gold[int(pos[1])].read().decode("utf-8")
  g2 = gold[int(pos[1])].read().decode("utf-8")
  selected_test = t1 + t2
  selected_gold = g1 + g2
else:
  selected_test = test[int(pos[0])].read().decode("utf-8")
  selected_gold = gold[int(pos[0])].read().decode("utf-8")

training_split = selected_training.split('\n'+'\n')
test_split = selected_test.split('\n'+'\n')
gold_split = selected_gold.split('\n')
dict_split = selected_dict.split('\n'+'\n')


# cleaning data:

clean_training = []
clean_test =  []
clean_dict = []

for item in training_split:
  if(len(item)>0):
    item = item.lower()
    tag = str(re.findall('<tag.+">',item))
    item = re.sub('<tag.+</>', '', item)
    item = re.sub('[^a-zA-Z]+', ' ', item)
    clean_words = [] 
    for word in item.split(' '):
      if(word not in sw and len(word)>1):
        clean_words.append(word)
    clean_training.append((tag[8:14],clean_words))
    
for item in test_split:
  if(len(item)>0):
    item = item.lower()
    tag = str(item[0:6])
    item = re.sub('<tag.+</>', '', item)
    item = re.sub('[^a-zA-Z]+', ' ', item)
    clean_words = [] 
    for word in item.split(' '):
      if(word not in sw and len(word)>1):
        clean_words.append(word)
    clean_test.append((tag, clean_words))
    
for item in dict_split:
  IDs = re.findall('<sen uid=.+', item)
  for defn in IDs:
    if(re.search('tag',defn) != None):
      spl = defn.split(' ')
      for s in spl:
        if(s[0:3]=='uid' or s[0:3]=='tag'):
          clean_dict.append(s)

final_dict = defaultdict(str)
for i in range(0,len(clean_dict),2):
  uid = clean_dict[i][4:]
  tag = clean_dict[i+1][4:]
  if(tag[-1]=='>'):
    tag = tag[0:-1]
  else:
    tag = tag 
  final_dict[uid] = tag
  
clean_gold = defaultdict(str)
for item in gold_split:
  uid = item[0:6]
  tag = item[7:]
  clean_gold[uid] = tag
  

###############################################################################
    
def getClassProbs(training, smooth = 1):
  class_counts = defaultdict(int)   
  class_probs = defaultdict(float)
  for item in training:
    class_counts[item[0]]+=1
  n = sum(class_counts.values())
  for k,v in class_counts.items():
    class_probs[k] = (v+smooth)/(n+(smooth*len(class_counts)))
  return class_probs


def getFeatProbs(training, test, smooth = 1):
  feat_probs = defaultdict(lambda: defaultdict(float))
  feat_counts = defaultdict(lambda: defaultdict(int))
  vocab = []
  
  for item in training:
    for word in item[1]:
      vocab.append(word)
      
  for item in test:
    for word in item[1]:
      vocab.append(word)
      
  vocab = set(vocab)
    
  for item in training:
    for word in item[1]:
      feat_counts[item[0]][word]+=1
      
  for c in feat_counts:
    for word in vocab:
      if word not in feat_counts[c]:
          feat_counts[c][word] = 0    
  
  for c in feat_counts:
    n = sum(feat_counts[c].values())
    for k,v in feat_counts[c].items():
      feat_probs[c][k] = (feat_counts[c][k]+smooth)/(n+smooth*len(set(vocab)))
  return feat_probs  

###############################################################################

classProbDict = getClassProbs(clean_training, smooth = 1) 
featProbDict = getFeatProbs(clean_training, clean_test) 

count = 0
correct = 0

for item in clean_test:
  count+=1
  tag = item[0]; bag_of_words = item[1]
  if(tag.isdigit() == True and len(bag_of_words)>0):
    score_dict = defaultdict(float)
    for c in classProbDict:
      prod = 1
      for word in bag_of_words:
        prod = prod*featProbDict[c][word]
      prod = prod*classProbDict[c]
      score_dict[c] = prod
    ID = max(score_dict.items(), key=operator.itemgetter(1))[0]
    dec = final_dict[ID]
    true_id = clean_gold[tag]
    if(dec==true_id):
      correct+=1
  else:
    print() 
  
print(str((correct/count)*100)+'%')




  

  


  
