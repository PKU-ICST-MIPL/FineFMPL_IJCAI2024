import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


template = ['a photo of a {}, a type of bird.']


class CUB(DatasetBase):

    dataset_dir = '/home/sunhongbo/DATA/FCIL_data/CUB_200_2011/'

    def __init__(self, root, num_shots, txt):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.test_images_txt = os.path.join(self.dataset_dir, 'test_images.txt')
            
        self.template = template



        classnames = []
        with open(os.path.join(self.dataset_dir, 'classes.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')[1]
                line = line.split('.')[1]
                line = line.replace('_', ' ')

                classnames.append(line)

        cname2lab = {c: i for i, c in enumerate(classnames)}







        filepath = os.path.join(self.dataset_dir, txt)
        with open(filepath, 'r') as f: 
            lines = f.readlines()
            last_line = lines[-1].strip()

        label_num = last_line.split('.')[0]

        res = []
        with open(self.test_images_txt, 'r') as f: 
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                num = line.split('.')[0]
                if int(num)<=int(label_num):
                    res.append(line)
            
        temp_test = 'temp_test'+txt.split('_')[1]+'.txt'
        temp_test_filepath = os.path.join(self.dataset_dir, temp_test)

        with open(temp_test_filepath, 'w') as f: 
            for i in range(len(res)):
                f.write(res[i])
                f.write('\n')

        if txt.split('_')[1].split('.')[0] == '1':
            train = self.read_data(cname2lab, txt)
        else:
            train = self.read_data(cname2lab, txt)
            #train = self.read_data(cname2lab, txt, flag=1, class_num=int(label_num))


        val = self.read_data(cname2lab, temp_test)
        test = self.read_data(cname2lab, temp_test)
        

        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, split_file, flag=0, class_num=0):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                
                imname = line.strip()
                classname = line.split('/')[0]
                classname = classname.split('.')[1]
                classname = classname.replace('_', ' ')

                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]

                if flag==1:
                    label = label - class_num

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items