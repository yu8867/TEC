import subprocess
from T5Tools.PredictAccuracy import test_data_predict

if __name__ == '__main__':
    subprocess.run(["python", "/home/user/TEC/source/T5Model/T5Learning.py", "1"])
    subprocess.run(["python", "/home/user/TEC/source/T5Model/T5Learning.py", "0"])

    print("true")
    test_data_predict(True)
    print("false")
    test_data_predict(False)