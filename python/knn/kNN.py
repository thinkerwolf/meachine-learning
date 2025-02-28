import numpy as np
import operator
import matplotlib
import os
import matplotlib.pyplot as plt


def create_data_set():
    """
    Creates a dataset and corresponding labels.
        :return:
            group (ndarray): The dataset, a 2D array where each row represents a data point.
            labels (list): The labels corresponding to each data point in the dataset.
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    Classifies a data point based on the k nearest neighbors.
    """
    dataSetSize = dataSet.shape[0]
    # print(f"dataSet shape: {dataSet.shape}")
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # matrix
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


######
# 以下是约会网站的例子
######


def file2matrix(filename):
    """
    datingTestSet.txt
    玩游戏所耗时间百分比	每年获得的飞行常客里程数	每周消费的冰淇淋公升数	喜欢程度
    40920	             8.326976	             0.953952	         largeDoses
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        label = 0
        if "didntLike" == listFromLine[-1]:
            label = 1
        elif "smallDoses" == listFromLine[-1]:
            label = 2
        elif "largeDoses" == listFromLine[-1]:
            label = 3
        classLabelVector.append(label)
        index += 1
    return returnMat, classLabelVector


def plot_scatter(datingDataMat, datingLabels):
    """
    Plots a scatter plot of the dating data."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        datingDataMat[:, 1],
        datingDataMat[:, 2],
        s=15.0 * np.array(datingLabels),  # 设置尺寸
        c=15.0 * np.array(datingLabels),  # 设置颜色
    )
    plt.show()


def auto_norm(dataSet):
    """
    归一化特征值
    Normalizes the dataset.
    """
    minVals = dataSet.min(0)  # 从列中选取最小值
    maxVals = dataSet.max(0)  # 从列中选取最大值
    ranges = maxVals - minVals  # 求差
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]  # 行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # dataSet - 重复minVals m次
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # normDataSet / 重复ranges m次
    return normDataSet, ranges, minVals


def dating_class_test():
    """
    Tests the dating classifier.
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3
        )
        print(
            f"the classifier came back with: {classifierResult}, the real answer is: {datingLabels[i]}"
        )
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f"the total error rate is: {errorCount/float(numTestVecs)}")


def classify_person():
    """
    Classifies a person based on the dating data.
    """
    resultList = ["not at all", "in small doses", "in large doses"]
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = auto_norm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3
    )  # 预测参数也要归一化
    print(f"You will probably like this person: {resultList[classifierResult - 1]}")


######
# 以下是手写数字识别的例子
######
def img2vector(filename):
    """
    Converts an image to a vector.
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwriteClassTest():
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(f"trainingDigits/{fileNameStr}")

    testFileList = os.listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector(f"testDigits/{fileNameStr}")
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(
            f"the classifier came back with: {classifierResult}, the real answer is: {classNumStr}"
        )
        if classifierResult != classNumStr:
            errorCount += 1.0
    print(f"\nthe total number of errors is: {errorCount}")
    print(f"\nthe total error rate is: {errorCount/float(mTest)}")
