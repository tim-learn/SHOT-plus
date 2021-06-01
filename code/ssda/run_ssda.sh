#!/bin/sh

# id seed
for ((s=0;s<=3;s++))
do
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office-home --max_epoch 50 --s $s

    for ((t=0;t<=3;t++))
    do
        if [ $s -eq $t ];then
            echo "skipped"
        else
            echo "okay"
            python image_target.py --ssl 0.0 --cls_par 0.0 --ent '' --gpu_id $1 --s $s --t $t --output_src "ckps/s"$2 --output "ckps/st"$2 --seed $2 --dset office-home 
            python image_mixmatch.py --ps 0.0 --cls_par 0.0 --gpu_id $1 --s $s --t $t --output_tar "ckps/st"$2 --output "ckps/mm_st"$2 --seed $2 --dset office-home --max_epoch 50

            python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --t $t --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
            python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --t $t --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

            python image_target.py --ssl 0.2 --cls_par 0.1 --gpu_id $1 --s $s --t $t --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
            python image_mixmatch.py --ps 0.0 --ssl 0.2 --cls_par 0.1 --gpu_id $1 --s $s --t $t --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

        fi
    done
done