from datetime import datetime


def errorOpen(path):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    logFile = open(path+"/"+current_time+"_error.txt", "a")
    print("Open Error file --")
    return logFile

def logOpen(path):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    logFile = open(path+"/"+current_time+"_log.txt", "a")
    print("Open Log file --")
    return logFile


def writeLog(log,str):
    now = datetime.now()
    current_time = now.strftime("%Y/%m/%d %H/%M/%S")
    print("LOG: ",str)
    log.write("["+current_time+"] "+str+"\n")
