# MARSWIN
This code mainly solves the problem of Mars MoRIC image super resolution.

### Training
To train marswin, run the following commands. You may need to change the `dataroot_H`, `dataroot_L`, `scale factor`, etc. in the json file for different settings. 



```python

python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_psnr.py --opt options/marswin/train_marswin_sr_realworld_x2_psnr.json  --dist True

```

You can also train above models using `DataParallel` as follows, but it will be slower.




## Reference
The code is based on 
1. KAIR (https://github.com/cszn/KAIR)
2. Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN)