import pandas as pd
import os
import argparse
import sys
import daymaker_train as train 

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str,
                    default="{0}/RadiologyFeedback_meghana/Daymakers/savedModel/".format(os.getcwd()), 
                    help='path to the trained model')

parser.add_argument('--traindata', type=str, default="{0}/Daymaker_train.xlsx".format(os.getcwd()),
                    help='path to the train .xlsx file that contains comments with labels ')

parser.add_argument('--validationdata', type=str, default="{0}/Daymaker_test.xlsx".format(os.getcwd()), 
                    help='path to the validation file .xlsx that contains comments with labels; only needed for training')
 
parser.add_argument('--testdata', type=str, default="{0}/Daymaker_test.xlsx".format(os.getcwd()), 
                    help='path to the test file .xlsx that contains comments without labels; only needed for testing ')
# parser.add_argument('--savepath', type=str, default="./Test_Student.xlsx", 
#                     help='path to save the annotated test file .xlsx that contains comments and model derieved labels; only needed for testing ')
parser.add_argument('--flag', type=str, default='Test', 
                    help='flag to signify the mode of model use -  Train/Test')

args = parser.parse_args()


def main():
    feedback = train.radiologyretive()

    # df = pd.read_excel(args.data)
    # # validdf = pd.read_excel(args.validationdata)
    # # labels = list(traindf)
    # # labels.remove('Comments')
    # feedback.train_main(df)
    # # feedback.model_save(args.modelpath) 
    # feedback.automate()
    # # print('Saved the data to: '+args.savepath)
    if args.flag == 'Train':
        traindf = pd.read_excel(args.traindata)
        validdf = pd.read_excel(args.validationdata) 
        feedback.train_main(traindf, validdf)
        feedback.model_save(args.modelpath)
        feedback.train_automate() 
            
    if args.flag == 'Test':
        testdf = pd.read_excel(args.testdata)
        feedback.model_load(args.modelpath)
        annotated_test = feedback.test_main(testdf,args.modelpath)
        print('finished the comments categorization')
        feedback.test_automate()
        print('finished generating template') 

if __name__ == "__main__":
    main()