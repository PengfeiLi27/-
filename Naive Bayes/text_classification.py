import numpy as np
class Naive_Bayes:
    # create distinct word list
    def _creatwordList(self,dataSet):
        """
        input:    dataSet = [[a,b,k],[b,g,h,k]]
        output:   return = [a,b,g,h,k]
        """
        vocabSet = set([])  
        for document in dataSet:
            vocabSet = vocabSet | set(document)  
        return list(vocabSet)
    
    # create word vector
    def _bagOfsetOfWordToVec(self,wordList, inputSet):
        """
        input:    wordList = [a,b,c,d,e]
                  inputSet = [a,c,c,k]
                  
        output:   return = [1,0,2,0,0]
        """

        returnVec = [0] * len(wordList)
        for word in inputSet:
            if word in wordList:
                returnVec[wordList.index(word)] += 1
        return returnVec
    
    # 
    def _trainNB0(self,trainMatrix, trainCategory):
        """
        input:    trainMatrix = [[1,0,0,1,1],
                                 [0,1,2,0,0],
                                  ...
                                 [0,0,1,0,0]]
                  trainCategory = [1,0,1,...]
                  
        output:   p0Vec = [P(w_1|+),P(w_2|+),P(w_3|+),P(w_4|+),P(w_5|+)]
                  p1Vec = [P(w_1|-),P(w_2|-),P(w_3|-),P(w_4|-),P(w_5|-)]
                  p_Negative = P(-)
        """
        # number of train sample
        numTrainDocs = len(trainMatrix)
        # number of word for each sample
        numWords = len(trainMatrix[0])
        # compute P(-)
        p_Negative = sum(trainCategory) / float(numTrainDocs)
        
        # intialize p(w|+) p(w|-)
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        
        # Laplace smooth
        p0Denom = 2.0
        p1Denom = 2.0
        
        for i in range(numTrainDocs):
            if trainCategory[i] == 0:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
            else:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
        p1Vect = np.log(p1Num / p1Denom)
        p0Vect = np.log(p0Num / p0Denom)
        return p0Vect, p1Vect, p_Negative
    
    # compute argmax P(v_i) * P(w_1|v_i) * ... * P(w_n|v_i)
    def _classifyNB(self,vec2Classify, p0Vec, p1Vec, p_Negative):
        """
        input:    vec2Classify = [1,0,2,0,0]
                  p0Vec = [P(w_1|+),P(w_2|+),P(w_3|+),P(w_4|+),P(w_5|+)]
                  p1Vec = [P(w_1|-),P(w_2|-),P(w_3|-),P(w_4|-),P(w_5|-)]
                  p_Negative = P(-)
                  
        output:   1 or 0
        """
        p1 = sum(vec2Classify * p1Vec) + np.log(p_Negative)
        p0 = sum(vec2Classify * p0Vec) + np.log(1 - p_Negative)
        if p1 > p0:
            return 1
        else:
            return 0
        
    # predict test 
    def predict(self,listOPosts, listClasses, testSample):
        output = []
        
        wordList = self._creatwordList(listOPosts)
       
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(self._bagOfsetOfWordToVec(wordList, postinDoc))
        p0V, p1V, pAb = self._trainNB0(trainMat, listClasses)
       
        for item in testSample:
            thisDoc = np.array(self._bagOfsetOfWordToVec(wordList, item))
            result=self._classifyNB(thisDoc, p0V, p1V, pAb)
            output.append(result)
            
        return output

# Main function
if __name__=="__main__":
    model = Naive_Bayes()
    
    X_train = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    
    # 1: negative    0: not negative
    y_train = [0, 1, 0, 1, 0, 1]
    
    X_test = [['love', 'my', 'girl', 'friend'],
                 ['stupid', 'garbage'],
                 ['Haha', 'I', 'really', "Love", "You"],
                 ['This', 'is', "my", "dog"],
                 ['maybe','stupid','worthless']]
    
    y_predict = model.predict(X_train, y_train, X_test)
    
    print(y_predict)
    
    