import csv
def findAvg(HA, round, team, Rows):
    for row in Rows:
        if row[1] == round and row[0] == team:
            if HA == 'H':
                return row[2]
            else:
                return row[3]

def findPD(team, Rows):
    for row in Rows:
        if row[0] == team:
            return row[1]

with open('laliga1718_ds1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "HomeTeam", "AwayTeam", "avgH", "avgA", "PDH", "PDA", "FTHG", "FTAG", "OvUn2.5"])
    with open('avgGoalsScored.csv', 'r', newline='') as file2:
        reader = csv.reader(file2)
        Rows1 = []
        for row in reader:
            Rows1.append(row)
    with open('powerDefence.csv', 'r', newline='') as file4:
        reader = csv.reader(file4)
        Rows2 = []
        for row in reader:
            Rows2.append(row)
    with open('laliga1718.csv', 'r', newline='') as file3:
        reader2 = csv.reader(file3)
        count = 0
        for row in reader2:
            if count == 0:
                count += 1
                continue
            if int(row[0]) < 4:
                continue
            Round = row[0]
            HomeTeam = row[2]
            AwayTeam = row[3]
            FTHG = int(row[4])
            FTAG = int(row[5])
            if FTHG + FTAG > 2.5:
                OvUn25 = 'Over'
            else:
                OvUn25 = 'Under'
            avgH = findAvg('H', Round, HomeTeam, Rows1)
            avgA = findAvg('A', Round, AwayTeam, Rows1)
            PDH = findPD(HomeTeam, Rows2)
            PDA = findPD(AwayTeam, Rows2)
            writer.writerow([Round, HomeTeam, AwayTeam, avgH, avgA, PDH, PDA, FTHG, FTAG, OvUn25])
