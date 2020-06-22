
GPU=0
CINNAMON_DATA_PATH=/media/D/ADL2020-SPRING/project/cinnamon/

###########################################################################
###########################   Wu's scripts   ##############################

############## naive baseline ##############
DELTA=7 #5
wu_train_naive_baseline:
	python3 ./naive_baseline/main.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--model naive --lr 1e-5 --epoch 60 --batch-size 4 --num-workers 8 --delta $(DELTA) --decline-lr \
		--save-path ./naive_baseline/ckpt/naive/ 
wu_inference_naive_baseline_dev:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--load-model ./naive_baseline/ckpt/naive/epoch_60.pt\
		--ref-file /media/D/ADL2020-SPRING/project/cinnamon/dev/dev_ref.csv\
		--dev_or_test dev \
		--model naive --delta $(DELTA) \
		--score --postprocess
wu_inference_naive_baseline_test:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--load-model ./naive_baseline/ckpt/naive/epoch_60.pt\
		--dev_or_test test\
		--model naive --delta $(DELTA)\
		--postprocess

############## blstm #################
DELTA2=5
wu_train_blstm:
	python3 ./naive_baseline/main.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--model blstm --lr 1e-5 --epoch 60 --batch-size 4 --num-workers 8 --delta $(DELTA2)\
		--save-path ./naive_baseline/ckpt/blstm/
wu_inference_blstm_dev:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--load-model ./naive_baseline/ckpt/blstm/epoch_60.pt\
		--ref-file /media/D/ADL2020-SPRING/project/cinnamon/dev/dev_ref.csv\
		--dev_or_test dev\
		--model blstm --delta $(DELTA2)\
		--score --postprocess 
wu_inference_blstm_test:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH)\
		--load-model ./naive_baseline/ckpt/blstm/epoch_60.pt\
		--dev_or_test test\
		--model blstm --delta $(DELTA2)\
		--postprocess 

###########################################################################
###########################   Hsu's scripts   #############################

hsu_train_naive_baseline:
	python3 ./naive_baseline/main.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH) --lr 2e-5 --epoch 8 --batch-size 4 --save-path ./naive_baseline/ckpt/ 

hsu_inference_naive_baseline:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --load-model ./naive_baseline/ckpt/epoch_35.pt --cinnamon-data-path $(CINNAMON_DATA_PATH) 



