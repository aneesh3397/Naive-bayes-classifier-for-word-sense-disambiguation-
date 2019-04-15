# Naive-bayes-classifier-for-word-sense-disambiguation-

As the title suggests, this project explored using the naive bayes classifier to perform word sense disambiguation. An example of the problem of word sense disambiguation is as follows:

1) There are a lot of banks along the river. 
2) There are a lot of banks in the city. 

In sentence 1), the 'banks' in question are likely river banks. In 2), 'banks' likely refers to the financial institution. It is an important task in NLP to be able to deal with ambiguity. 

The naive bayes approach to dealing with ambiguity involves treating the words surrounding the target word (in this case, bank) as features that can be used to classify the target into its correct sense. For instance, when the word bank is intended to mean 'river bank', we are likely to see words such as 'river', 'flow', 'nature', 'woods' etc. For the financial institution, we are more likely to see monetary terms such as 'savings', 'money', 'interest' etc. Given enough training data, the model will use the words surrounding the target word in a test case, such as:

3) The banks are closed today. 

to hopefully disambiguate 'banks' correctly. 


The code here deals with data from the SENSEVAL data set. I have included the training and test data as well as a dictionary that contains the meanings of all the tags (that serve as classes for a given word). There is also a 'gold standard' file that is used for evaluation of the model. 

The model is set up to take in the users input to select one of 29 words from the SENSEVAL data set. Once the user has made their choice, the model is trained on the training data for that word and then is tested on the test data for that word. The model then prints the percentage of test cases that were classified correctly. Performance of the model varied greatly depending on the word, ranging from 30% at worst to about 75% at best. 
