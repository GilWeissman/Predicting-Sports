import csv

def Average(lst):
    print(lst)
    return sum(lst) / len(lst)

def calc_avg(home_lst, away_lst):
    lenH = len(home_lst)
    lenA = len(away_lst)
    sumH = sum(home_lst)
    sumA = sum(away_lst)
    if lenH == 3:
        avgH = Average(home_lst)
        avgA = Average(home_lst)
    if lenH == 2:
        avgH = (2 * sumH + sumA) / 5
        avgA = (sumH + 2 * sumA) / 4
    if lenH == 1:
        avgH = (2 * sumH + sumA) / 4
        avgA = (sumH + 2 * sumA) / 5
    if (lenH) == 0:
        avgH = Average(away_lst)
        avgA = Average(away_lst)
    return avgH, avgA


with open('Malaga.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "avgHome", "avgAway"])
    with open('laliga1718.csv', 'r', newline='') as file2:
        reader = csv.reader(file2)
        Rows = []
        for row in reader:
            Rows.append(row)
    for round in range(4, 39):
        print(round)
        home = []
        away = []
        countRow = 0
        for row in Rows:
            if countRow != 0 and int(row[0]) < round and int(row[0]) > round - 4 and \
                    (row[2] == 'Malaga' or row[3] == 'Malaga'):
                if row[2] == 'Malaga':
                    home.append(int(row[4]))
                else:
                    away.append(int(row[5]))
            countRow += 1
        avgH, avgA = calc_avg(home, away)
        writer.writerow([round, avgH, avgA])

    # writer.writerow([1, "Linus Torvalds", "Linux Kernel"])
    # writer.writerow([2, "Tim Berners-Lee", "World Wide Web"])
    # writer.writerow([3, "Guido van Rossum", "Python Programming"])
