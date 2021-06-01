# !/bin/sh


for ((s=0;s<=3;s++))
do
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office-caltech --net resnet101 --max_epoch 100 --s $s
    python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset office-caltech --net resnet101 --max_epoch 100

    python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-caltech --net resnet101
    python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-caltech --net resnet101 --max_epoch 100

    python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-caltech --net resnet101
    python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-caltech --net resnet101 --max_epoch 100

done


for ((s=0;s<=3;s++))
do
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset office-home --net resnet50 --max_epoch 50 --s $s
    python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --net resnet50 --max_epoch 50

    python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home --net resnet50
    python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --net resnet50 --max_epoch 50

    python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset office-home --net resnet50
    python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset office-home --net resnet50 --max_epoch 50

done


for ((s=0;s<=3;s++))
do
    python image_source.py --gpu_id $1 --seed $2 --output "ckps/s"$2 --dset pacs --net resnet18 --max_epoch 100 --s $s
    python image_mixmatch.py --ps 0.0 --cls_par 0.0 --model source --gpu_id $1 --s $s --output_tar "ckps/s"$2 --output "ckps/mm"$2 --seed $2 --dset pacs --net resnet18 --max_epoch 100

    python image_target.py --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset pacs --net resnet18
    python image_mixmatch.py --ps 0.0 --ssl 0.0 --cls_par 0.0 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset pacs --net resnet18 --max_epoch 100

    python image_target.py --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_src "ckps/s"$2 --output "ckps/t"$2 --seed $2 --dset pacs --net resnet18
    python image_mixmatch.py --ps 0.0 --ssl 0.6 --cls_par 0.3 --gpu_id $1 --s $s --output_tar "ckps/t"$2 --output "ckps/mm"$2 --seed $2 --dset pacs --net resnet18 --max_epoch 100

done




for ((y=2019;y<=2021;y++))
do
    for ((t=0;t<=3;t++))
    do
        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset office-caltech --t $t --cls_par 0.0 --ssl 0.0
        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset office-caltech --t $t --cls_par 0.3 --ssl 0.6

        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset pacs --net resnet18 --t $t --cls_par 0.0 --ssl 0.0
        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset pacs --net resnet18 --t $t --cls_par 0.3 --ssl 0.6

        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset office-home --net resnet50 --t $t --cls_par 0.0 --ssl 0.0
        python image_ms.py --output "san_ms/s"$y --output_src "ckps/s"$y --output_tar "ckps/t"$y --output_mm "ckps/mm"$y --dset office-home --net resnet50 --t $t --cls_par 0.3 --ssl 0.6

    done
done