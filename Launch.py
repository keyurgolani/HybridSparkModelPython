import findspark
findspark.init()
import Utils

from HybridModel import HybridModel

def main():
    with open("input.txt", "r+") as raw_data:
        data = map(lambda x: x.replace("\n", "").split("\t"), list(raw_data))
        
        # data = map(lambda x: x[:17]+x[18:]+[x[17] if x[17] == '1' else 0], data)
        # data = map(Utils.vectorize, data)
        # test, train = Utils.randomSplit([0.5, 0.5], data)
        # model = HybridModel(train)
        # model.run_test(test)
        # print model.accuracy()


if __name__ == '__main__':
    main()