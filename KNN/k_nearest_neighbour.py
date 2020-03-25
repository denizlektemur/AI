import numpy as np

def scale(data, min_data, max_data):
    difference = max_data - min_data
    data -= min_data
    data = float(data)/float(difference)
    return data

def get_distance(day_1, day_2):
    return (pow(day_1[0] - day_2[0], 2)) + (pow(day_1[1] - day_2[1], 2)) + (pow(day_1[2] - day_2[2], 2))\
         + (pow(day_1[3] - day_2[3], 2)) + (pow(day_1[4] - day_2[4], 2)) + (pow(day_1[5] - day_2[5], 2))\
         + (pow(day_1[6] - day_2[6], 2))

def percentage_overlap(list1, list2):
    amount_overlap = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            amount_overlap += 1
    return amount_overlap/len(list1)*100

data = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5:
lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
labels = []
for label in dates:
    if label < 20000301:
        labels.append("winter")
    elif 20000301 <= label < 20000601:
        labels.append("lente")
    elif 20000601 <= label < 20000901:
        labels.append("zomer")
    elif 20000901 <= label < 20001201:
        labels.append("herfst")
    else: # from 01-12 to end of year
        labels.append("winter")

validation_data = np.genfromtxt("validation1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5:
lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validation_dates = np.genfromtxt("validation1.csv", delimiter=";", usecols=[0])
validation_labels = []

for label in validation_dates:
    if label < 20010301:
        validation_labels.append("winter")
    elif 20010301 <= label < 20010601:
        validation_labels.append("lente")
    elif 20010601 <= label < 20010901:
        validation_labels.append("zomer")
    elif 20010901 <= label < 20011201:
        validation_labels.append("herfst")
    else: # from 01-12 to end of year
        validation_labels.append("winter")

FG = []
TG = []
TN = []
TX = []
SQ = []
DR = []
RH = []

for dataset in data:
    FG.append(dataset[0])
    TG.append(dataset[1])
    TN.append(dataset[2])
    TX.append(dataset[3])
    SQ.append(dataset[4])
    DR.append(dataset[5])
    RH.append(dataset[6])

FG_min = min(FG)
FG_max = max(FG)
TG_min = min(TG)
TG_max = max(TG)
TN_min = min(TN)
TN_max = max(TN)
TX_min = min(TX)
TX_max = max(TX)
SQ_min = min(SQ)
SQ_max = max(SQ)
DR_min = min(DR)
DR_max = max(DR)
RH_min = min(RH)
RH_max = max(RH)

scaled_data = data

for dataset in scaled_data:
    dataset[0] = scale(dataset[0], FG_min, FG_max)
    dataset[1] = scale(dataset[1], TG_min, TG_max)
    dataset[2] = scale(dataset[2], TN_min, TN_max)
    dataset[3] = scale(dataset[3], TX_min, TX_max)
    dataset[4] = scale(dataset[4], SQ_min, SQ_max)
    dataset[5] = scale(dataset[5], DR_min, DR_max)
    dataset[6] = scale(dataset[6], RH_min, RH_max)

for dataset in validation_data:
    dataset[0] = scale(dataset[0], FG_min, FG_max)
    dataset[1] = scale(dataset[1], TG_min, TG_max)
    dataset[2] = scale(dataset[2], TN_min, TN_max)
    dataset[3] = scale(dataset[3], TX_min, TX_max)
    dataset[4] = scale(dataset[4], SQ_min, SQ_max)
    dataset[5] = scale(dataset[5], DR_min, DR_max)
    dataset[6] = scale(dataset[6], RH_min, RH_max)


k = 100


found_labels = []
for validation_day in range(len(validation_data)):
    found_labels.append([])
    distances = []
    for i in range(len(scaled_data)):
        distances.append([get_distance(validation_data[validation_day], scaled_data[i]), i])
    distances.sort()
    nearest_labels = []
    for distance in distances:
        nearest_labels.append(labels[distance[1]])
    for current_k in range(1, k+1):
        amount_winter = 0
        amount_lente = 0
        amount_zomer = 0
        amount_herfst = 0
        for j in range(current_k):
            if nearest_labels[j] == "winter":
                amount_winter += 1
            elif nearest_labels[j] == "lente":
                amount_lente += 1
            elif nearest_labels[j] == "zomer":
                amount_zomer += 1
            elif nearest_labels[j] == "herfst":
                amount_herfst += 1
        if amount_winter > amount_lente and amount_winter > amount_zomer and amount_winter > amount_herfst:
            found_labels[validation_day].append("winter")
        elif amount_lente > amount_winter and amount_lente > amount_zomer and amount_lente > amount_herfst:
            found_labels[validation_day].append("lente")
        elif amount_herfst > amount_lente and amount_herfst > amount_zomer and amount_herfst > amount_winter:
            found_labels[validation_day].append("herfst")
        elif amount_zomer > amount_lente and amount_zomer > amount_winter and amount_zomer > amount_herfst:
            found_labels[validation_day].append("zomer")
        else:
            found_labels[validation_day].append(nearest_labels[0])

k_percentages = []

for i in range(len(found_labels[0])):
    found_validation_labels = []
    for j in range(len(found_labels)):
        found_validation_labels.append(found_labels[j][i])
    k_percentages.append([percentage_overlap(found_validation_labels, validation_labels), i+1])

k_percentages.sort()
print("Best K is", k_percentages[-1][1], "with a percentage of:", str(k_percentages[-1][0])+"%")