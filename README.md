# Contrastive Learning of Visual Representations Using Video Frame Pairs
Welcome to my project for CMSC 31230 at the University of Chicago. Please make yourself at home.

## Instructions
1. Clone repository
   ````
	git clone https://github.com/lawrenceztang/simclr-2
	cd simclr-2
   ````

3. Install dependencies
 	````
	conda install --file requirements.txt
	pip install -U openmim
	mim install mmcv
	````

5. Download Kinetic Dataset
 	````
	git clone https://github.com/cvdfoundation/kinetics-dataset.git
	cd kinetics-dataset
	bash ./k400_downloader.sh
	bash ./k400_extractor.sh
	````
7. Run and cancel
	````
	python run.py --train_mode=pretrain --train_batch_size=512 --train_epochs=0 --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 --dataset=paired --image_size=32 --eval_split=test --resnet_depth=18 --use_blur=False --color_jitter_strength=0.5 --model_dir=/tmp/simclr_test --use_tpu=False --data_dir=paired_dataset
	````
8. Extract image pairs
 	````
	python3 create_dataset.py
	````
10. Run model
	````
 	python run.py --train_mode=pretrain --train_batch_size=512 --train_epochs=82 --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 --dataset=paired --image_size=32 --eval_split=test --resnet_depth=18 --use_blur=False --color_jitter_strength=0.5 --model_dir=/tmp/simclr_test --use_tpu=False --data_dir=paired_dataset
	````
11. Finetune
	````
	python run.py --mode=train_then_eval --train_mode=finetune \
	--fine_tune_after_block=4 --zero_init_logits_layer=True \
	--variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
	--global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
	--train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
	--dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
	--checkpoint=/tmp/simclr_test --model_dir=/tmp/simclr_test_ft --use_tpu=False
	````
13. View on Tensorboard
	````
	python -m tensorboard.main --logdir=/tmp/simclr_test_ft
	````

