import sys
from sklearn import metrics
from sklearn.metrics.classification import confusion_matrix


def eval(keys, predictions):
    """ Simple minded eval system: file1 is gold standard, file2 is system outputs. Result is straight accuracy. """

    count = 0.0
    correct = 0.0

    for key, prediction in zip(keys, predictions):
        key = key.strip()
        prediction = prediction.strip()
        if key == '': continue
        count += 1
        if key == prediction: correct += 1
    print("Evaluated ", count, " tags.")
    print("Accuracy is: ", correct / count)

def cm(keys,predictions):
    keyslist =  zip(keys)
    predlist =  zip(predictions)
    cm = confusion_matrix(keyslist, predlist)
    print(cm)
    
    
if __name__ == "__main__":
    keys = open(sys.argv[1])
    print(keys)
    predictions = open(sys.argv[2])
    eval(keys, predictions)
  #  cm(keys,predictions)
   # print(confusion_matrix)
  