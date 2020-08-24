
import datetime
import math
import re

from testPackage.Predictions import Rain

classifier = Rain()

dataFile = open("2017-weather.txt",'r')
for line in dataFile:
    x = re.findall("-\d+\.\d+|\d+\.\d+|\d+|[A-Z]",line)
    year = re.match("[0-9]{4}",x[1])
    year = int(year.group(0))
    month = re.match("[0-9]{4}([0-9]{2})",x[1])
    month = int(month.group(1))
    day = re.match("[0-9]{4}[0-9]{2}([0-9]{2})",x[1])
    day = int(day.group(1))
    monthDay = re.match("[0-9]{4}([0-9]{4})",x[1])
    date = datetime.datetime.strptime(monthDay.group(1),"%m%d")
    dayOfYear = int(date.timetuple().tm_yday)-1
    sin = math.sin(dayOfYear)
    cos = math.cos(dayOfYear)
    hour = int(x[2])
    latitude = float(x[7])
    longitude = float(x[6])
    temperature = float(x[8])
    #input the data
    if(float(x[9]) > 0.0):
        classifier.input_train_data(year, month, day, hour, sin, cos, latitude, longitude, temperature, "Rain")
    else:
        classifier.input_train_data(year, month, day, hour, sin, cos, latitude, longitude, temperature, "No Rain")
        
dataFile.close()

#train model
classifier.train()


#process the testing data
dataFile = open("2018-weather.txt",'r')
for line in dataFile:
    x =re.findall("-\d+\.\d+|\d+\.\d+|\d+|[A-Z]",line)
    year = re.match("[0-9]{4}",x[1])
    year = int(year.group(0))
    month = re.match("[0-9]{4}([0-9]{2})",x[1])
    month = int(month.group(1))
    day = re.match("[0-9]{4}[0-9]{2}([0-9]{2})",x[1])
    monthDay = re.match("[0-9]{4}([0-9]{4})",x[1])
    date = datetime.datetime.strptime(monthDay.group(1),"%m%d")
    dayOfYear = int(date.timetuple().tm_yday)-1
    sin = math.sin(dayOfYear)
    cos = math.cos(dayOfYear)
    day = int(day.group(1))
    hour = int(x[2])
    latitude = float(x[7])
    longitude = float(x[6])
    temperature = float(x[8])
    #input the data
    if float(x[9]) > 0.0:
        classifier.input_test_data(year, month, day, hour, sin, cos, latitude, longitude, temperature)
        classifier.label("Rain")
    else:
        classifier.input_test_data(year, month, day, hour, sin, cos, latitude, longitude, temperature)
        classifier.label("No Rain")
        
dataFile.close()



# output prediction or probability

#print("Prediction",classifier.predict())
print("Prediction Pobability",classifier.predict_prob())



# output accuracy

#classifier.accuracy_score()


# find out which features are useful
#classifier.useful_features()
