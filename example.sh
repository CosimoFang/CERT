export PRETRAINED_MODEL_PATH=./moco_model/moco.p
export GLUE_DIR=./data/CoLA
export BERT_ALL_DIR=./    
export TASK=cola
export OUTPUT_PATH=mocoout

python train.py \
    --task_name $TASK \
        --do_train --do_val --do_test --do_val_history \
	    --do_save \
	        --do_lower_case \
		    --bert_model bert-large-uncased \
		        --bert_load_path $PRETRAINED_MODEL_PATH \
			    --bert_load_mode model_only \
			        --bert_save_mode model_all \
				    --train_batch_size 16 \
				        --learning_rate 2e-5 \
					    --output_dir $OUTPUT_PATH


export OUTPUT_PATH=mocoout1  
python train.py \
    --task_name $TASK \
        --do_train --do_val --do_test --do_val_history \
            --do_save \
                --do_lower_case \
                    --bert_model bert-large-uncased \
                        --bert_load_path $PRETRAINED_MODEL_PATH \
                            --bert_load_mode model_only \
                                --bert_save_mode model_all \
                                    --train_batch_size 16 \
                                        --learning_rate 3e-5 \
                                            --output_dir $OUTPUT_PATH
export OUTPUT_PATH=mocoout2
python train.py \
    --task_name $TASK \
        --do_train --do_val --do_test --do_val_history \
            --do_save \
                --do_lower_case \
                    --bert_model bert-large-uncased \
                        --bert_load_path $PRETRAINED_MODEL_PATH \
                            --bert_load_mode model_only \
                                --bert_save_mode model_all \
                                    --train_batch_size 16 \
                                        --learning_rate 4e-5 \
                                            --output_dir $OUTPUT_PATH

export OUTPUT_PATH=mocoout3  
python train.py \
    --task_name $TASK \
        --do_train --do_val --do_test --do_val_history \
            --do_save \
                --do_lower_case \
                    --bert_model bert-large-uncased \
                        --bert_load_path $PRETRAINED_MODEL_PATH \
                            --bert_load_mode model_only \
                                --bert_save_mode model_all \
                                    --train_batch_size 16 \
                                        --learning_rate 5e-5 \
                                            --output_dir $OUTPUT_PATH








					    















					    
