import svmMLiA

dataArr , labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

print labelArr
print b

print alphas[alphas>0]

    shape(alphas[alphas>0])