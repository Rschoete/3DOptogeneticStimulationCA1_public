# this script crops the batch simulation list based on the log with succesful completes
# Results is to do list with uncompleted simulations
folder = 'SDC20221011_invitroCA1pyrs5/'
completeInputlist = folder+'runlistopt.csv'
completed_log = folder+'wsubrun.pbs.log52028414'

with open(completeInputlist,'r') as f:
    inputsToDO = [line.rstrip('\n') for line in f]
with open(completed_log,'r') as f:
    mylog = [line.rstrip('\n') for line in f]

columnheader = inputsToDO.pop(0)
mylog_c1 = [x.split(' ',1)[0] for x in mylog]
counts = [mylog_c1.count(str(i+1)) for i in range(len(inputsToDO))]
todo_indx = [i for i,x in enumerate(counts) if x==1]

inputsToDO2 = inputsToDO.copy()

inputsToDO = [inputsToDO[idx] for idx in todo_indx]



print(len(inputsToDO), len(todo_indx))

inputsToDO = inputsToDO
print(inputsToDO)
with open(folder+'inputsToDO.csv','w') as file:
    file.write(columnheader+'\n')
    for  item in inputsToDO:
        file.write("%s\n" % item)

