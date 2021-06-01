#!/bin/sh

# id seed

for ((s=0;s<=2;s++))
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office --max_epoch 100 --s $s
    python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset office --max_epoch 100

    python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office
    python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office --max_epoch 100

    python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office 
    python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office --max_epoch 100
done


for ((s=0;s<=3;s++))
do
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office-home --max_epoch 50 --s $s
    python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

    python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
    python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50

    python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home 
    python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --max_epoch 50
done

python image_source.py --gpu_id $1 --seed $2 --output "ckps/res101/s"$2 --dset VISDA-C --max_epoch 10 --s 0 --net resnet101
python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s 0 --t 1 --output_tar "ckps/res101/s"$2 --output "ckps/res101/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10 --net resnet101

python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_src "ckps/res101/s"$2 --output "ckps/res101/t"$2 --seed $2 --dset VISDA-C  --net resnet101
python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/res101/t"$2 --output "ckps/res101/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10 --net resnet101

python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_src "ckps/res101/s"$2 --output "ckps/res101/t"$2 --seed $2 --dset VISDA-C  --net resnet101
python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/res101/t"$2 --output "ckps/res101/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10 --net resnet101







python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset VISDA-C --max_epoch 10 --s 0
python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s 0 --t 1 --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

python image_target.py --gent '' --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
python image_mixmatch.py --gent '' --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

python image_target.py --ssl 0.0 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.3 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10

python image_target.py --ssl 0.6 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset VISDA-C 
python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.0 --gpu_id $1 --s 0 --t 1 --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset VISDA-C --max_epoch 10
