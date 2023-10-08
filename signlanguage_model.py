import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

def load_data():
	imagepath = "Sign-Language-Digits-Dataset/Dataset"
	digit_folders = ["0","1","2","3","4","5","6","7","8","9"]

	X = [] # features
	y = [] # labels

	# import the image-data
	for folder in digit_folders:
		for filename in glob.glob(imagepath + "/"+folder + "/*.jpg"):
			img = Image.open(filename)
			grayscale_img = ImageOps.grayscale(img)
			npImg = np.asarray(grayscale_img)

			X.append(npImg)
			y.append(int(folder))

	X = np.asarray(X).reshape(len(X), 100*100) # transform X into a numpy array and reshape the images

	return X,y


def pca_init(X, N):
	pca = PCA(n_components=N)
	pca.fit(X)

	return pca


def train_model(X,y, K):
	clf = KNeighborsClassifier(n_neighbors=K)
	scores = cross_val_score(clf, X, y, cv=10)
	mean_score = sum(scores)/10
	#print("Model accuracy:", mean_score)

	clf.fit(X, y)

	return clf,mean_score

# method for generating and visualisizing the confusion matrix for our model
def generate_confusion_matrix(X,y):
	y_preds = []
	y_tests = []
	for _ in range(10):
		# we need to split the data in order to get an accurate confusion matrix
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
		clf = train_model(X_train,y_train, 10)[0]
		y_pred = clf.predict(X_test)

		y_preds.append(y_pred)
		y_tests.append(y_test)

	y_preds_final = []
	y_tests_final = []
	for i in range(len(y_preds)):
		for j in range(len(y_preds[i])):
			y_preds_final.append(y_preds[i][j])
			y_tests_final.append(y_tests[i][j])

    # visualize the confusion matrix
	ax = plt.subplot()
	c_mat = confusion_matrix(y_tests_final, y_preds_final)
	sns.heatmap(c_mat, annot=True, fmt='g', ax=ax)

	ax.set_xlabel('Predicted labels', fontsize=15)
	ax.set_ylabel('True labels', fontsize=15)
	ax.set_title('Confusion Matrix', fontsize=15)

	plt.show()

# method for plotting the model accuracy with different pca N values
def plot_pca_accuracy(X):
	scores = []
	max_N = 200
	for N in range(20, max_N):
		pca = pca_init(X, N)
		X_reduced = pca.transform(X)
		scores.append(train_model(X_reduced, y, 10)[1])
	plt.plot(range(20,max_N),scores, color="r")

	plt.title("Model accuracy with different pca-N-values")
	plt.xlabel("N-value")
	plt.ylabel("Accuracy")
	plt.xticks(range(20,max_N,10))	
	plt.yticks(np.arange( round(min(scores),2), round(max(scores),2), 0.01 ))	# set y-axis ticks to be from the smallest acc to biggest with 0.01 increments
	plt.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.7)		# Add a customized grid to the background

	plt.show()


# method for plotting the model accuracy with different values of K
def plot_k_accuracy(X,y):
	scores = []
	max_k = 50
	# train model with different k-values and save the k-fold cross-validation -accuracy scores
	for k in range(1,max_k):
		scores.append(train_model(X,y, K=k)[1])

	plt.plot(range(1,max_k),scores, color="r")
	plt.title("Model accuracy with different K-values")
	plt.xlabel("K-value")
	plt.ylabel("Accuracy")
	plt.xticks(range(1,max_k))
	plt.yticks(np.arange( round(min(scores),2), round(max(scores),2), 0.01 ))	# set y-axis ticks to be from the smallest acc to biggest with 0.01 increments
	plt.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.7)		# Add a customized grid to the background

	plt.show()


N = 70
K = 10
if __name__ == "__main__":
	X,y = load_data()
	pca = pca_init(X, N)
	X_reduced = pca.transform(X)

	clf = train_model(X_reduced,y, K)[0]

	#plot_pca_accuracy(X)
	#generate_confusion_matrix(X_reduced,y)
	#plot_k_accuracy(X_reduced,y)