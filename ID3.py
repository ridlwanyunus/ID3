import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# dataset = np.array([
#     ['Sunny','Hot','High','Weak','No'],
#     ['Sunny','Hot','High','Strong','No'],
#     ['Overcast','Hot','High','Weak','Yes'],
#     ['Rain','Mild','High','Weak','Yes'],
#     ['Rain','Cool','Normal','Weak','Yes'],
#     ['Rain','Cool','Normal','Strong','No'],
#     ['Overcast','Cool','Normal','Strong','Yes'],
#     ['Sunny','Mild','High','Weak','No'],
#     ['Sunny','Cool','Normal','Weak','Yes'],
#     ['Rain','Mild','Normal','Weak','Yes'],
#     ['Sunny','Mild','Normal','Strong','Yes'],
#     ['Overcast','Mild','High','Strong','Yes'],
#     ['Overcast','Hot','Normal','Weak','Yes'],
#     ['Rain','Mild','High','Strong','No']
# ])
#
# dataset_columns = ['Outlook','Temperature','Humidity','Wind','Play Tennis']
# train_data_m = pd.DataFrame(data=dataset, columns=dataset_columns)

#train_data_m = pd.read_csv("train\PlayTennis.csv")
#train_data_m.head()

# print(train_data_m)
class ID3:
    train_data_m = None
    label = None

    def __init__(self, d, f):
        self.train_data_m = d
        self.label = f

    def calc_total_entropy(self, train_data, label, class_list):
        total_row = train_data.shape[0]  # the total size of the dataset
        total_entr = 0

        for c in class_list:  # for each class in the label
            total_class_count = train_data[train_data[label] == c].shape[0]  # number of the class
            total_class_entr = - (total_class_count / total_row) * np.log2(
                total_class_count / total_row)  # entropy of the class
            total_entr += total_class_entr  # adding the class entropy to the total entropy of the dataset

        return total_entr


    def calc_entropy(self, feature_value_data, label, class_list):
        class_count = feature_value_data.shape[0]
        entropy = 0

        for c in class_list:
            label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]  # row count of class c
            entropy_class = 0
            if label_class_count != 0:
                probability_class = label_class_count / class_count  # probability of the class
                entropy_class = - probability_class * np.log2(probability_class)  # entropy
            entropy += entropy_class
        return entropy


    def calc_info_gain(self, feature_name, train_data, label, class_list):
        feature_value_list = train_data[feature_name].unique()  # unqiue values of the feature
        total_row = train_data.shape[0]
        feature_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = train_data[
                train_data[feature_name] == feature_value]  # filtering rows with that feature_value
            feature_value_count = feature_value_data.shape[0]
            feature_value_entropy = self.calc_entropy(feature_value_data, label,
                                                 class_list)  # calculcating entropy for the feature value
            feature_value_probability = feature_value_count / total_row
            feature_info += feature_value_probability * feature_value_entropy  # calculating information of the feature value

        return self.calc_total_entropy(train_data, label,
                                  class_list) - feature_info  # calculating information gain by subtracting


    def find_most_informative_feature(self, train_data, label, class_list):
        feature_list = train_data.columns.drop(label)  # finding the feature names in the dataset
        # N.B. label is not a feature, so dropping it
        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:  # for each feature in the dataset
            feature_info_gain = self.calc_info_gain(feature, train_data, label, class_list)
            if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature


    def generate_sub_tree(self, feature_name, train_data, label, class_list):
        feature_value_count_dict = train_data[feature_name].value_counts(
            sort=False)  # dictionary of the count of unqiue feature value
        tree = {}  # sub tree or node

        for feature_value, count in feature_value_count_dict.iteritems():
            feature_value_data = train_data[
                train_data[feature_name] == feature_value]  # dataset with only feature_name = feature_value

            assigned_to_node = False  # flag for tracking feature_value is pure class or not
            for c in class_list:  # for each class
                class_count = feature_value_data[feature_value_data[label] == c].shape[0]  # count of class c

                if class_count == count:  # count of (feature_value = count) of class (pure class)
                    tree[feature_value] = c  # adding node to the tree
                    train_data = train_data[train_data[feature_name] != feature_value]  # removing rows with feature_value
                    assigned_to_node = True
            if not assigned_to_node:  # not pure class
                tree[feature_value] = "?"  # as feature_value is not a pure class, it should be expanded further,
                # so the branch is marking with ?

        return tree, train_data


    def make_tree(self, root, prev_feature_value, train_data, label, class_list):
        if train_data.shape[0] != 0:  # if dataset becomes enpty after updating
            max_info_feature = self.find_most_informative_feature(train_data, label, class_list)  # most informative feature
            tree, train_data = self.generate_sub_tree(max_info_feature, train_data, label,
                                                 class_list)  # getting tree node and updated dataset
            next_root = None

            if prev_feature_value != None:  # add to intermediate node of the tree
                root[prev_feature_value] = dict()
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:  # add to root of the tree
                root[max_info_feature] = tree
                next_root = root[max_info_feature]

            for node, branch in list(next_root.items()):  # iterating the tree node
                if branch == "?":  # if it is expandable
                    feature_value_data = train_data[train_data[max_info_feature] == node]  # using the updated dataset
                    self.make_tree(next_root, node, feature_value_data, label, class_list)  # recursive call with updated dataset


    def decision_tree(self):
        print('ID3 Decision Tree')
        train_data = self.train_data_m.copy() #getting a copy of the dataset
        tree = {} #tree which will be updated
        class_list = train_data[self.label].unique() #getting unqiue classes of the label
        self.make_tree(tree, None, train_data, self.label, class_list) #start calling recursion
        return tree



    # print(tree)
    # print(tree['Outlook']['Rain']['Wind']['Weak'])

    # datatest = np.array([
    #     ['Rain','Hot','High','Strong',''],
    # ])

    # test_data_m = pd.DataFrame(data=datatest, columns=dataset_columns)

    def predict(self, tree, instance):
        if not isinstance(tree, dict): #if it is leaf node
            return tree #return the value
        else:
            root_node = next(iter(tree)) #getting first key/feature name of the dictionary
            feature_value = instance[root_node] #value of the feature
            if feature_value in tree[root_node]: #checking the feature value in current tree node
                return self.predict(tree[root_node][feature_value], instance) #goto next feature
            else:
                return None

    def evaluate(self, tree, test_data_m, label):
        correct_preditct = 0
        wrong_preditct = 0
        for index, row in test_data_m.iterrows(): #for each row in the dataset
            result = self.predict(tree, test_data_m.iloc[index]) #predict the row
            print(test_data_m.iloc[index])
            print(result)
            if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
                correct_preditct += 1 #increase correct count
            else:
                wrong_preditct += 1 #increase incorrect count
        accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
        return accuracy

#
# id3 = ID3(train_data_m, 'Play Tennis')
# tree = id3.decision_tree()
# print(tree)