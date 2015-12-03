import matplotlib.pyplot as plt

from sklearn import svm, datasets

digits = datasets.load_digits()

clf = svm.SVC(gamma =0.001, C =100)

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print ('prediction:',clf.predict(digits.data[:-3]))

plt.imshow(digits.images[-3],cmap=plt.cm.gray_r, interpolation="nearest")

plt.show()