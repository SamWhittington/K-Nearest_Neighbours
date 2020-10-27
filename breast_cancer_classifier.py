import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

## Form of data
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)

#print(breast_cancer_data.target, breast_cancer_data.target_names)

## Splitting into training and validation sets - using sklearns train_test_split function here

# Setting the variables as:
training_set, validation_set, training_labels, validation_labels = train_test_split(
  breast_cancer_data.data,
  breast_cancer_data.target,
  test_size=0.2,
  random_state=100
)

## Sanity check that data is same length:
#print(len(training_set), len(training_labels))

## The classifier (from sklearn.neighbors)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training_set, training_labels)

## Checking against validation data:
## Validation is ~94% when k=3
#print(classifier.score(validation_set, validation_labels))

## Finding a better k-value
for k in range(1,100):
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_set, training_labels)
  #print(classifier.score(validation_set, validation_labels))

## Graphing the above results (with pyplot as plt)
k_list = range(1,101)
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_set, training_labels)
  accuracies.append(classifier.score(validation_set, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Data')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
