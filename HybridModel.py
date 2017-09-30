import pyspark
import Utils
sc = pyspark.SparkContext(appName="HybridSparkModel")
from pyspark.mllib import classification
from pyspark.mllib import tree
from pyspark.mllib import regression

class HybridModel():
    def __init__(self, train, confidence_threshold=0.5):
        self.train = train
        self.test = []
        self.confidence_threshold = confidence_threshold
        self.ambiguousTPs = [];
        self.confidentTPs = [];
        self.ambiguousTNs = [];
        self.confidentTNs = [];
        self.ambiguousFPs = [];
        self.confidentFPs = [];
        self.ambiguousFNs = [];
        self.confidentFNs = [];
        
        self.maliciousSamples = [];
        self.benignSamples = [];

        train_rdd = sc.parallelize(map(lambda x: regression.LabeledPoint(x[-1], x[:-1]), self.train))
        self.models = [
            classification.LogisticRegressionWithSGD.train(data=train_rdd, iterations=10, step=1.0, miniBatchFraction=1.0, initialWeights=None, regParam=0.01, regType='l2', intercept=False, validateData=True, convergenceTol=0.001),
            classification.LogisticRegressionWithLBFGS.train(data=train_rdd, iterations=10, initialWeights=None, regParam=0.01, regType='l2', intercept=False, corrections=10, tolerance=0.0001, validateData=True, numClasses=2),
            classification.SVMWithSGD.train(data=train_rdd, iterations=10, step=1.0, regParam=0.01, miniBatchFraction=1.0, initialWeights=None, regType='l2', intercept=False, validateData=True, convergenceTol=0.001),
            classification.NaiveBayes.train(data=train_rdd, lambda_=1.0),
            tree.DecisionTree.trainClassifier(data=train_rdd, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0),
            tree.RandomForest.trainClassifier(data=train_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=3, featureSubsetStrategy='auto', impurity='gini', maxDepth=4, maxBins=32, seed=None),
            tree.GradientBoostedTrees.trainClassifier(data=train_rdd, categoricalFeaturesInfo={}, loss='logLoss', numIterations=10, learningRate=0.1, maxDepth=3, maxBins=32)
        ]

    def run_test(self, test):
        self.test = test
        for data_point in self.test:
            decisions = map(lambda model: model.predict(data_point[:-1]), self.models)
            prediction = Utils.mode(decisions)
            confidence = float(decisions.count(prediction)) / len(decisions)
            print decisions, prediction, confidence, data_point[-1]
            if confidence > self.confidence_threshold:
                if prediction == data_point[-1]:
                    if prediction == 1:
                        self.confidentTNs.append(data_point)
                    else:
                        self.confidentTPs.append(data_point)
                else:
                    if prediction == 1:
                        self.confidentFNs.append(data_point)
                    else:
                        self.confidentFPs.append(data_point)
            else:
                if prediction == data_point[-1]:
                    if prediction == 1:
                        self.ambiguousTNs.append(data_point)
                    else:
                        self.ambiguousTPs.append(data_point)
                else:
                    if prediction == 1:
                        self.ambiguousFNs.append(data_point)
                    else:
                        self.ambiguousFPs.append(data_point)

    
    def ambiguous(self):
        return self.ambiguousFNs + self.ambiguousFPs + self.ambiguousTNs + self.ambiguousTPs

    def confident(self):
        return self.confidentFNs + self.confidentFPs + self.confidentTNs + self.confidentTPs

    def TPs(self):
        return self.ambiguousTPs + self.confidentTPs

    def TNs(self):
        return self.ambiguousTNs + self.confidentTNs

    def FPs(self):
        return self.ambiguousFPs + self.confidentTPs

    def FNs(self):
        return self.ambiguousFNs + self.confidentFNs

    def malicious(self):
        return self.TPs() + self.FNs()

    def benign(self):
        return self.FPs() + self.TNs()

    def accuracy(self):
        return {
            "TPR" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FNs()))), 
            "TNR" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FPs()))), 
            "PPV" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FPs()))), 
            "NPV" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FNs()))), 
            "FPR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TNs()))), 
            "FNR" : float(len(self.FNs())) / (float(len(self.FNs())) + float(len(self.TPs()))), 
            "FDR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TPs()))), 
            "ACC" : (float(len(self.TPs())) + float(len(self.TNs()))) / (float(len(self.TPs())) + float(len(self.TNs())) + float(len(self.FPs())) + float(len(self.FNs()))), 
            "F1" : (2 * float(len(self.TPs()))) / (2 * float(len(self.TPs())) + float(len(self.FPs())) + float(len(self.FNs())))
        }
