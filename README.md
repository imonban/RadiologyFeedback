# RadiologyFeedback

Model categorizes patients comments from the radiology department in different buckets. 

Create conda enviromnet.
conda create --name 'Provide a new name' --file requirements.txt

## Train
python main.py --flag Train --traindata 'file path' --validationdata 'file path'

## Test
python main.py --flag Test --testdata 'Excel file name' --savepath 'Save file path'

Contact Banerjee.Imon@mayo.edu for any issues
