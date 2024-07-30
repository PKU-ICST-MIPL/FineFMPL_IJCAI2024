import os
import time    


start_time = time.time()

#command_name = ['oxford_pets.yaml','fgvc.yaml','imagenet.yaml','sun397.yaml','food101.yaml','caltech101.yaml','dtd.yaml','eurosat.yaml','oxford_flowers.yaml','stanford_cars.yaml','ucf101.yaml']
command_name = ['cifar100.yaml']

for name in command_name:
    command = 'sh ./conduct.sh'+' '+name
    os.system(command)

print('total time: {:.1f}min'.format((time.time()-start_time)/60))