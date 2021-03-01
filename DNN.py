from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from numpy import random
import math
from scipy import stats
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



digits = datasets.load_digits() #Loads the digit dataset
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.3) # Creates the 4 different variables whilst randomly stelctingg which are chosen for test and train


def dnnRunWithConv(x_train, y_train, x_test, y_test):

    img_rows, img_cols = 8, 8

    x_trainRS = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_testRS = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_trainRS /= 255
    x_testRS /= 255


    num_classes = 10

    y_trainCat = to_categorical(y_train, num_classes)
    y_testCat = to_categorical(y_test, num_classes)



    dnnConvModel = Sequential()
    dnnConvModel.add(Conv2D(16, kernel_size=(3, 3),
        activation='relu',
        input_shape=(img_rows, img_cols, 1)))

    dnnConvModel.add(Conv2D(32, (3, 3), activation='relu'))
    dnnConvModel.add(MaxPooling2D(pool_size=(2, 2)))

    dnnConvModel.add(Dropout(0.25))

    dnnConvModel.add(Flatten())

    dnnConvModel.add(Dense(128, activation='relu'))
    dnnConvModel.add(Dropout(0.5))
    dnnConvModel.add(Dense(num_classes, activation='softmax'))

    dnnConvModel.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    batch_size = 10
    epochs = 128

    dnnConvModel.fit(x_trainRS, y_trainCat,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_testRS, y_testCat))
    score = dnnConvModel.evaluate(x_testRS, y_testCat, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    dnnConvModel.save("test_model.h5")
    return(score[1])

def dnnRunNonConv(x_train, y_train, x_test, y_test):

    img_rows, img_cols = 8, 8
    num_classes = 10

    dnnnoConvModel = Sequential()
    dnnnoConvModel.add(Dropout(0.25))
    dnnnoConvModel.add(Flatten())
    dnnnoConvModel.add(Dense(128, activation='relu'))
    dnnnoConvModel.add(Dropout(0.5))
    dnnnoConvModel.add(Dense(num_classes, activation='softmax'))

    dnnnoConvModel.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    batch_size = 10
    epochs = 128

    dnnnoConvModel.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = dnnnoConvModel.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    dnnnoConvModel.save("test_model.h5")
    return(score[1])


def chooseXAmount(lengthList, usedVal,chosenArr,chosenArrYvalues,values,valuesY):
    i=0
    while (i<math.floor(lengthList/5)):
        x = random.randint(lengthList)
        if doesContain(usedVal, x) == False:
            chosenArr.append(values[x])
            chosenArrYvalues.append(valuesY[x])
            usedVal.append(x)
            i+=1

def doesContain (araa, checkint):
    contains = False
    for z in range (len(araa)):
        if (araa[z]==checkint):
            contains = True
    return contains

def changeToNumPy(listToChangeX,listToChangeY):
    listToChangeX = np.asarray(listToChangeX, dtype=np.float32)
    listToChangeY = np.asarray(listToChangeY, dtype=np.float32)
    return(listToChangeX,listToChangeY)

def euclidian_distance(row1,row2): #Calculates the euclidian distance using the given mathematical formula
    distance = 0.0
    for i in range (len(row1)):
        distance += (row1[i]-row2[i])**2
    return np.sqrt(distance)

def sort(listToSort): #A quicksort algorithm that sorts both the list of distances but also their indexes so that their y values can be retrieved at a later date
    n = len(listToSort)
    indexArray = makeArray(n)
    for i in range (n):
        sorted = True
        for j in range (n - i - 1):
            if listToSort[j] > listToSort[j+1]:
                listToSort[j], listToSort[j + 1] = listToSort[j + 1], listToSort[j]
                indexArray[j], indexArray[j + 1] = indexArray[j + 1], indexArray[j]

                sorted = False
        if sorted:
            break
    return listToSort, indexArray

def makeArray(n): #Makes a list that is n long going from 1 to n
    numberlist = []
    for i in range (n):
        numberlist.append(i)
    return numberlist

def choosefinal(xvalues, chosenkvalue,yvalues): #Takes the first k values in the index list and creates a list using them that contains their given y values
    value_indexes = (xvalues[1])[0:chosenkvalue]
    choseny = []
    for i in range (len(value_indexes)):
        choseny.append(yvalues[value_indexes[i]])
    return choseny

def findDigit(n, X_train, Y_train, X_test, Y_test): #Given a single set of values will find what digit they represent
    distances = []
    testvalue = X_test[n]
    testy = Y_test[n]

    for i in range (len(X_train)): #Calculates the euclidian distance for every value in thr training data
        
        distances.append(euclidian_distance(X_train[i],testvalue))

    sortedDistances = sort(distances) #Sorts the distances from smallest to largest

    finalarray = (choosefinal(sortedDistances, 5, Y_train)) #Grabs the y values for k closest distances

    chosenY = (((stats.mode(finalarray))[0])[0]) #Calculates the mode of the data in the array of digits

    if (testy==chosenY): #If the computers prediction was correct will return True else returns False
        found=True
    else:
        found=False
    return found,chosenY

def ownMadeKnn(X_train, Y_train, X_test, Y_test, showCounter): #Loops through all the data points keeping track of when the computer was right and when the computer was wrong
    y_pred = np.array([], dtype='int16')
    right = 0.0
    wrong = 0.0
    for i in range (len(X_test)):
        print("hi")
        if showCounter == "Y":
            print(i)
        found,pred = findDigit(i, X_train, Y_train, X_test, Y_test)
        y_pred = np.append(y_pred , pred)
        if (found == True):
            right += 1
        else:
            wrong += 1
    percentage = ((right/(wrong+right))*100) #Calculate's the percentage accuracy of the algorithm
    errorRate = 100.0-percentage #Calculate's the error rate of the algorithm
    return round(errorRate,2),y_pred

def builtInKnn(X_train, y_train, X_test, y_test, chosenKDistance): #Runs the build in SKLearn algorithm
    knn = KNeighborsClassifier(n_neighbors=chosenKDistance)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knnLibraryAccuracy = (metrics.accuracy_score(y_test, y_pred) * 100)
    errorRate = 100.0 - knnLibraryAccuracy
    return (round(errorRate,2),y_pred)

def makeAccuracy(err):
    return(1-(err/100))

def crossVal():
    totalaccuracyNonConv = 0.0
    totalaccuracyWithConv = 0.0
    totalaccuracyImpKNN = 0.0
    totalaccuracyOwnKNN = 0.0

    values = digits.data
    valuesY = digits.target
    usedVal = []
    array1 = []
    array1y = []
    array2 = []
    array2y = []
    array3 = []
    array3y = []
    array4 = []
    array4y = []
    array5 = []
    array5y = []
    lengthList = len(values)
    
    chooseXAmount(lengthList, usedVal, array1,array1y,values,valuesY)
    chooseXAmount(lengthList, usedVal, array2,array2y,values,valuesY)
    chooseXAmount(lengthList, usedVal, array3,array3y,values,valuesY)
    chooseXAmount(lengthList, usedVal, array4,array4y,values,valuesY)
    chooseXAmount(lengthList, usedVal, array5,array5y,values,valuesY)

    onetestlist = array2+array3+array4+array5
    twotestlist = array1+array3+array4+array5
    threetestlist = array2+array1+array4+array5
    fourtestlist = array2+array3+array1+array5
    fivetestlist = array2+array3+array1+array4

    onetestlistY = array2y+array3y+array4y+array5y
    twotestlistY = array1y+array3y+array4y+array5y
    threetestlistY = array2y+array1y+array4y+array5y
    fourtestlistY = array2y+array3y+array1y+array5y
    fivetestlistY = array2y+array3y+array1y+array4y
    
    array1, array1y = changeToNumPy(array1,array1y)
    array2, array2y = changeToNumPy(array2,array2y)
    array3, array3y = changeToNumPy(array3,array3y)
    array4, array4y = changeToNumPy(array4,array4y)
    array5, array5y = changeToNumPy(array5,array5y)

    onetestlist, onetestlistY = changeToNumPy(onetestlist, onetestlistY)
    twotestlist, twotestlistY = changeToNumPy(twotestlist, twotestlistY)
    threetestlist, threetestlistY = changeToNumPy(threetestlist, threetestlistY)
    fourtestlist, fourtestlistY = changeToNumPy(fourtestlist, fourtestlistY)
    fivetestlist, fivetestlistY = changeToNumPy(fivetestlist, fivetestlistY)
    
    totalaccuracyNonConv += dnnRunNonConv(array1, array1y, onetestlist, onetestlistY)
    totalaccuracyNonConv += dnnRunNonConv(array2, array2y, twotestlist, twotestlistY)
    totalaccuracyNonConv += dnnRunNonConv(array3, array3y, threetestlist, threetestlistY)
    totalaccuracyNonConv += dnnRunNonConv(array4, array4y, fourtestlist, fourtestlistY)
    totalaccuracyNonConv += dnnRunNonConv(array5, array5y, fivetestlist, fivetestlistY)

    totalaccuracyWithConv += dnnRunWithConv(array1, array1y, onetestlist, onetestlistY)
    totalaccuracyWithConv += dnnRunWithConv(array2, array2y, twotestlist, twotestlistY)
    totalaccuracyWithConv += dnnRunWithConv(array3, array3y, threetestlist, threetestlistY)
    totalaccuracyWithConv += dnnRunWithConv(array4, array4y, fourtestlist, fourtestlistY)
    totalaccuracyWithConv += dnnRunWithConv(array5, array5y, fivetestlist, fivetestlistY)

    knnError = (builtInKnn(onetestlist, onetestlistY, array1, array1y, 5))[0]
    totalaccuracyImpKNN += makeAccuracy(knnError)
    knnError = (builtInKnn(twotestlist, twotestlistY, array2, array2y, 5))[0]
    totalaccuracyImpKNN += makeAccuracy(knnError)
    knnError = (builtInKnn(threetestlist, threetestlistY, array3, array3y, 5))[0]
    totalaccuracyImpKNN += makeAccuracy(knnError)
    knnError = (builtInKnn(fourtestlist, fourtestlistY, array4, array4y, 5))[0]
    totalaccuracyImpKNN += makeAccuracy(knnError)
    knnError = (builtInKnn(fivetestlist, fivetestlistY, array5, array5y, 5))[0]
    totalaccuracyImpKNN += makeAccuracy(knnError)

    knnError = (ownMadeKnn(onetestlist, onetestlistY, array1, array1y, "Y"))[0]
    totalaccuracyOwnKNN += makeAccuracy(knnError)
    knnError = (ownMadeKnn(twotestlist, twotestlistY, array2, array2y, "Y"))[0]
    totalaccuracyOwnKNN += makeAccuracy(knnError)
    knnError = (ownMadeKnn(threetestlist, threetestlistY, array3, array3y, "Y"))[0]
    totalaccuracyOwnKNN += makeAccuracy(knnError)
    knnError = (ownMadeKnn(fourtestlist, fourtestlistY, array4, array4y, "Y"))[0]
    totalaccuracyOwnKNN += makeAccuracy(knnError)
    knnError = (ownMadeKnn(fivetestlist, fivetestlistY, array5, array5y, "Y"))[0]
    totalaccuracyOwnKNN += makeAccuracy(knnError)

    print("Mean Accuracy without Convolution Layer " + str(totalaccuracyNonConv/5))
    print("Mean Accuracy with Convolution Layer " + str(totalaccuracyWithConv/5))
    print("Mean Accuracy with imported KNN " + str(totalaccuracyImpKNN/5))
    print("Mean Accuracy with imported KNN " + str(totalaccuracyImpKNN/5))

def ConfusionMatrix():
    model = keras.models.load_model('H:/Comp219/Work try/test_model.h5')
    x = model.predict(x_test)
    print(x)

ConfusionMatrix()
# dnnRunNonConv(x_train, y_train, x_test, y_test)