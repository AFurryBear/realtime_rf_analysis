from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ChanGainsIM

def load_ephyData(path, tStart, tEnd, chanList):
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window
    binFullPath = Path(path)
    root.destroy()

    meta = readMeta(binFullPath)
    sRate = SampRate(meta)

    firstSamp = int(sRate*tStart)
    lastSamp = int(sRate*tEnd)
    rawData = makeMemMapRaw(binFullPath, meta)
    selectData = rawData[chanList, firstSamp:lastSamp+1]
    return selectData, int(sRate), meta

def load_ttlData(path, ttl_chan):
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window
    binFullPath = Path(path)
    root.destroy()

    meta = readMeta(binFullPath)
    sRate = SampRate(meta)
    
    rawData = makeMemMapRaw(binFullPath, meta)
    selectData = rawData[ttl_chan,]
    return selectData, int(sRate), meta