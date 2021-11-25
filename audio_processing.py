import operator
import sys
from aubio import source, pitch
import pandas as pd
def note(pitch):

    if pitch < 58 or pitch > 74 or abs(round(pitch)-pitch) > 0.25:
        return ""
    return str(round(pitch)-59)
        
  
if len(sys.argv) < 2:
    sys.exit(1)

filename = sys.argv[1]

downsample = 1
samplerate = 44100 // downsample
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

win_s = 2096 // downsample # fft size
hop_s = 512  // downsample # hop size

# Read audio file
s = source(filename, samplerate, hop_s)
samplerate = s.samplerate

tolerance = 0.8

pitch_o = pitch("yin", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []

# Identify the note for each sample and maintain the count of each note
count=0
notelist=['']
notecount=dict({'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0})  
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    detected_note=note(pitch)
    if detected_note!='':
        if detected_note in notecount:
            notecount[detected_note]+=1
            if notelist[-1]!=detected_note:
                notelist.append(detected_note)
        else:
            notecount[detected_note]=1
    pitches += [pitch]
    if read < hop_s: break
    
# Notes 1, 8 and 13 will always be a part of all Melakartas
notes=['1','8','13']
sortednotes=[]
for each in sorted(notecount.items(), key=lambda kv: kv[1]):
    sortednotes.append(each[0])
    
# Select two swaras from 1st swara set based on frequency of occurance
swaracnt=2
for each in sortednotes[::-1]:
    if each in ['2','3','4','5']:
        notes.append(each)
        swaracnt-=1
        if swaracnt==0:
            break

# Select two swaras from 2nd swara set based on frequency of occurance
swaracnt=2
for each in sortednotes[::-1]:
    if each in ['9','10','11','12']:
        notes.append(each)
        swaracnt-=1
        if swaracnt==0:
            break
            
# Select one swara from 3rd swara set based on frequency of occurance
if notecount['6']>notecount['7']:
    notes.append('6')
else:
    notes.append('7')
            
print(notes)

# Match the patten of notes with the melakarta notes table to identify the audio file's classification
data=pd.read_csv("notes\melakartas.csv")
swaras=['1','2','3','4','5','6','7','8','9','10','11','12','13']
for index in range(len(data)):
    yes=True
    for note in swaras:
        if (note in notes and data[note][index]==1) or (note not in notes and data[note][index]==0):
            pass
        else:
            yes=False
    if yes==True:
        print(data["Name"][index])
        break