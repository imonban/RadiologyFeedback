# RadiologyFeedback

Model categorizes patients comments from the radiology department in different buckets. 

## Create conda environment.
conda create --name 'Provide a new name' --file requirements.txt

## Train
python main.py --flag Train --traindata 'file path' --validationdata 'file path'
The file should have 'Comments' column

## Test
python main.py --flag Test --testdata 'Excel file name' --savepath 'Save file path'
The file should have 'Comments' column

## Output
Creates a new file with additional columns derived by the model

Contact Banerjee.Imon@mayo.edu for any issues
