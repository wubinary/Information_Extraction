

#######################################################################

## test 
#model=trained_model/epoch_2_model_loss_4.9602.pt
#model=abc

## best
model=trained_model/epoch_6_model_loss_5.5226.pt 

threshold=0.5
#threshold=0.1
#threshold=0.05

#######################################################################

train_1070:
	python3 main.py --load-model $(model) --batch-size 4 --gpu 1

train_1080ti_from_scratch:
	python3 main.py --epoch 8 --lr 2e-5 --gpu 0 

train_1080ti:
	python3 main.py --epoch 3 --lr 2e-5 --load-model $(model) --gpu 0 

inference:
	python3 inference.py --threshold $(threshold) --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 

evaluate:
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result.json ckip/data/ 


#######################################################################

test_threshold:  
	python3 inference.py --threshold 0.1 --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result_0.1.json ckip/data/
	python3 inference.py --threshold 0.3 --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result_0.3.json ckip/data/  
	python3 inference.py --threshold 0.5 --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result_0.5.json ckip/data/
	python3 inference.py --threshold 0.7 --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result_0.7.json ckip/data/ 
	python3 inference.py --threshold 0.9 --test-json /media/D/ADL2020-SPRING/A2/dev.json --load-model-path $(model) --write-file result/predict.json --num-workers 8 --gpu 1 --batch-size 48 
	python3 evaluate.py /media/D/ADL2020-SPRING/A2/dev.json result/predict.json result/result_0.9.json ckip/data/
 


