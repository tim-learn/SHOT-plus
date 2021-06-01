#!/bin/sh

python uda_digit.py --dset m2u --gpu_id $1 --seed $2 --cls_par 0.0 --output ckps_digits
python uda_digit.py --dset u2m --gpu_id $1 --seed $2 --cls_par 0.0 --output ckps_digits
python uda_digit.py --dset s2m --gpu_id $1 --seed $2 --cls_par 0.0 --output ckps_digits
																																						
python uda_digit.py --dset m2u --gpu_id $1 --seed $2 --cls_par 0.1 --ssl 0.2 --output ckps_digits
python uda_digit.py --dset u2m --gpu_id $1 --seed $2 --cls_par 0.1 --ssl 0.2 --output ckps_digits
python uda_digit.py --dset s2m --gpu_id $1 --seed $2 --cls_par 0.1 --ssl 0.2 --output ckps_digits

python digit_mixmatch.py --dset m2u --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --model source
python digit_mixmatch.py --dset u2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --model source
python digit_mixmatch.py --dset s2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --model source

python digit_mixmatch.py --dset m2u --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.0
python digit_mixmatch.py --dset u2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.0
python digit_mixmatch.py --dset s2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.0

python digit_mixmatch.py --dset m2u --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.1 --ssl 0.2
python digit_mixmatch.py --dset u2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.1 --ssl 0.2
python digit_mixmatch.py --dset s2m --gpu_id $1 --seed $2 --output ckps_mm --output_tar ckps_digits --cls_par 0.1 --ssl 0.2