import subprocess, os, fnmatch, time, datetime, argparse
from natsort import natsorted, ns

parser = argparse.ArgumentParser(description='PyTorch Super Res Example Batch Processer')
parser.add_argument('--image', type=str, required=True, help="Path of Image you want to process")
parser.add_argument('--output_filename', type=str, required=True, help="What you want the output files to be named(Just type the text before 1,2...png)")
opt = parser.parse_args()
average1, average2 = 0, 0
def ordinal(n):
    if 10 <= n % 100 < 20:
        return str(n) + "th"
    else:
        return str(n) + { 1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n%10,"th")
def secondsend(n):
    if int(n) == int(1):
        return "Second"
    else:
        return "Seconds"
epochs = []
listOfFiles = os.listdir('.')
pattern = "model_epoch*"
for entry in listOfFiles:
    if fnmatch.fnmatch(entry,pattern):
        epochs.append(entry)
epochs = natsorted(epochs, key=lambda y: y.lower())
time.sleep(5)
average = 0
for i in range(len(epochs)):
    starttime = time.time()
    print("I am on the", ordinal(i+1),"Epoch")
    subprocess.run(["python","super_resolve.py", "--input_image", opt.image, "--model" ,epochs[i], "--output_filename", str(opt.output_filename)+str(i+1)+".png"])
    endtime = time.time()
    timetaken=round(endtime-starttime,1)
    if int(timetaken) == int(1):
        Second="Second"
    else:
        Second ="Seconds" 
    average1 = average1+timetaken
    average2 = round(average1/(i+1), 1)
    ETASeconds = average2 * (len(epochs)+1)
    print("Outputted to",str(opt.output_filename)+str(i+1)+".png.","Operation took", timetaken, secondsend(timetaken)+". Average Time Taken:", average2, secondsend(average2)+". Estimated Time of Completion:", datetime.timedelta(seconds=ETASeconds))
    
