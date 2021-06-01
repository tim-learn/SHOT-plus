#!/bin/sh

python image_pretrained.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --seed $2 --output "ckps/s"$2
python image_pretrained.py --ssl 0.0 --cls_par 0.3 --gpu_id $1 --seed $2 --output "ckps/s"$2
python image_pretrained.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --seed $2 --output "ckps/s"$2

for ((s=0;s<=3;s++))
do
    python image_source.py --da pda --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office-home --max_epoch 50 --s $s
    python image_mixmatch.py --model source --da pda --ps 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

    python image_target.py --da pda --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
    python image_mixmatch.py --da pda --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

    python image_target.py --da pda --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
    python image_mixmatch.py --da pda --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

done


for ((s=0;s<=1;s++))
do
    python image_source.py --da pda --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset VISDA-C --max_epoch 10 --s $s
    python image_mixmatch.py --model source --da pda --ps 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

    python image_target.py --da pda --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
    python image_mixmatch.py --da pda --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

    python image_target.py --da pda --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
    python image_mixmatch.py --da pda --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

done