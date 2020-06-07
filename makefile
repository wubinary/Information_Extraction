
GPU=0
CINNAMON_DATA_PATH=/media/D/ADL2020-SPRING/project/cinnamon/ 

###########################################################################
###########################   Wu's scripts   ##############################

wu_train_naive_baseline:
	python3 ./naive_baseline/main.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH) --lr 2e-5 --epoch 8 --batch-size 4 --save-path ./naive_baseline/ckpt/ 

wu_inference_naive_baseline:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --load-model ./naive_baseline/ckpt/epoch_6.pt --cinnamon-data-path $(CINNAMON_DATA_PATH) 


###########################################################################
###########################   Hsu's scripts   #############################

hsu_train_naive_baseline:
	python3 ./naive_baseline/main.py --gpu $(GPU) --cinnamon-data-path $(CINNAMON_DATA_PATH) --lr 2e-5 --epoch 8 --batch-size 4 --save-path ./naive_baseline/ckpt/ 

hsu_inference_naive_baseline:
	python3 ./naive_baseline/inference.py --gpu $(GPU) --load-model ./naive_baseline/ckpt/epoch_6.pt --cinnamon-data-path $(CINNAMON_DATA_PATH) 
	

