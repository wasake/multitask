python -m cerberusdet.train \
--img 640 --batch 8 \
--data data/lightweight_dataset.yaml \
--weights pretrained/yolov8x_state_dict.pt \
--cfg cerberusdet/models/yolov8x_voc_obj365.yaml \
--hyp data/hyps/hyp.cerber-voc_obj365.yaml \
--name lw \
--device 0 \
--epochs 5 \
--mlflow-url localhost