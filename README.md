This is the code for paper: xxx.

# 1.prepare env
1.1.create env
		
		conda create --name=env python=3.7.5
		source activate env
1.2.install torch

		pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

1.3.install others
		
		pip install -r requirements_gpu.txt
		
# 2.prepare datasets & models

2.1.prepare datasets

download datasets from  https://www.dropbox.com/s/lzznin2hxt6rmft/SC-Ques.tar.gz?dl=0 and save in dir "./datasets/SC-Ques"
		
2.2.prepare models

download models from hugging face and save in dir "./pretrained_models"
		

# 3.train models

		cd examples
		sh train.sh
		
# 4.predict questions

		cd examples
		python predict_question.py bert bert.pkl model_dir
