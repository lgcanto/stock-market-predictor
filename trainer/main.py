import os
import pandas as pd
import subprocess
import glob

MAX_DAYS = 5
df_companies = pd.read_csv("../datasets/company-codes.csv")

for index, row in df_companies.iterrows():
  asset = row['code']
  for d in range(MAX_DAYS):
    interval = d + 1
    # configName = asset + "_2c_" + str(interval) + "d"
    configName = asset + "_3c_" + str(interval) + "d"
    fileName = "configs/" + configName + ".json"
    with open("templateconfig_XXXXX_Yc_Zd.json") as inputFile, open(fileName, "w") as outputFile:
      for line in inputFile:
        outputFile.write(line.replace("XXXXX_Yc_Zd", configName))

    print("Executing for " + configName)

    bashCommand = "pytext train < " + fileName
    result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
    result.stdout.decode('utf-8')

    print("Renaming run folder")

    runFoldersList = glob.glob("runs/*")
    latestRunFolder = max(runFoldersList, key=os.path.getctime)
    os.rename(latestRunFolder, "runs/" + configName)

    print("Execution done for " + configName)