import pandas as pd
import os
import argparse
import sys
import train
import train_action
# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str,
                    default="./model/", 
                    help='path to the trained model')
parser.add_argument('--traindata', type=str, default="./Train_Student.xlsx",
                    help='path to the train .xlsx file that contains comments with labels ')
parser.add_argument('--validationdata', type=str, default="./Test_Student.xlsx", 
                    help='path to the validation file .xlsx that contains comments with labels; only needed for training')
parser.add_argument('--testdata', type=str, default="./Test_Student.xlsx", 
                    help='path to the test file .xlsx that contains comments without labels; only needed for testing ')
parser.add_argument('--savepath', type=str, default="./Test_Student.xlsx", 
                    help='path to save the annotated test file .xlsx that contains comments and model derieved labels; only needed for testing ')
parser.add_argument('--flag', type=str, default='Test', 
                    help='flag to signify the mode of model use -  Train/Test')



# parse the arguments
args = parser.parse_args()


def main():
    feedback = train.radiologyretive()
    action = train_action.radiologyaction()
    if args.flag == 'Train':
        try:
            traindf = pd.read_excel(args.traindata)
            validdf = pd.read_excel(args.validationdata)
            labels = list(traindf)
            labels.remove('Comments')
            feedback.train_main(traindf, validdf, labels)
            feedback.model_save(args.modelpath)
            action.train_main(traindf, validdf, labels)
            action.model_save(args.modelpath)
        except:
            sys.exit('Enter the path for the correct .xlsx file or model saving path')
    if args.flag == 'Test':
        #try:
            testdf = pd.read_excel(args.testdata)
            feedback.model_load(args.modelpath)
            action.model_load(args.modelpath)
            annotated_test = feedback.test_main(testdf)
            print('finished the comments categorization')
            annotated_test_action = action.test_main(annotated_test)
            print('finished the action annotation')
            annotated_test_action.to_excel(args.savepath)
            print('Saved the data to: '+args.savepath)
        #except:
        #    sys.exit('Enter the path for the correct .xlsx file or model saving path')
      
    


if __name__ == "__main__":
    main()