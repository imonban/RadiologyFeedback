import pandas as pd
import os
import argparse
import sys
import train

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str,
                    default="./model/", 
                    help='path to the trained model')
#parser.add_argument('--data', type=str, default="/mnt/storage/User_Feedback/Data/Daymaker Categorization.xlsx",
#                    help='path to the train .xlsx file that contains comments with labels ')
parser.add_argument('--data', type=str, default="{0}/Daymaker Categorization.xlsx".format(os.getcwd()),
                    help='path to the train .xlsx file that contains comments with labels ')
# parser.add_argument('--savepath', type=str, default="./Test_Student.xlsx", 
#                     help='path to save the annotated test file .xlsx that contains comments and model derieved labels; only needed for testing ')


args = parser.parse_args()


def main():
    feedback = train.radiologyretive()

    df = pd.read_excel(args.data)
    # validdf = pd.read_excel(args.validationdata)
    # labels = list(traindf)
    # labels.remove('Comments')
    feedback.train_main(df)
    # feedback.model_save(args.modelpath) 
    feedback.automate()
    # print('Saved the data to: '+args.savepath)
        


if __name__ == "__main__":
    main()