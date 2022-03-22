import subprocess
from T5Tools.PredictAccuracy import test_data_predict

if __name__ == '__main__':

    print("true")
    test_data_predict(True)
    print("false")
    test_data_predict(False)