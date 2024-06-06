import os
import csv

class queens:
    def __init__(self):
        # Specify the paths to the CSV files
        self.queens_train_csv = "4queens_dataset_train.csv"
        self.queens_train_csv = "4queens_dataset_valid.csv"
        self.queens_test_csv = "4queens_dataset_test.csv"

        #Read data from CSV files
        #self.queens_train_data = self.read_csv(self.queens_train_csv)
        #self.queens_valid_data = self.read_csv(self.queens_train_csv)
        #self.queens_test_data = self.read_csv(self.queens_test_csv)
        
        def read_csv(fname):
            print(f"Reading {fname}...")
            with open(fname) as f:
                reader = csv.reader(f, delimiter=',')
                next(reader)
                return [(q, a) for q, a in reader]
        
        # self.queens_train = read_csv('4queens_dataset_train.csv')
        # #print ("Hi Train",self.queens_train)
        # self.queens_valid = read_csv('4queens_dataset_valid.csv')
        # #print ("Hi Valid",self.queens_valid)
        # self.queens_test = read_csv('4queens_dataset_test.csv')
        # #print ("HI Test",self.queens_test)
        
        self.train = read_csv('4queens_dataset_train.csv')
        #print ("Hi Train",self.queens_train)
        self.valid = read_csv('4queens_dataset_valid.csv')
        #print ("Hi Valid",self.queens_valid)
        self.test = read_csv('4queens_dataset_test.csv')
        #print ("HI Test",self.queens_test)


        

        
        

        