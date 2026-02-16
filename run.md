

Enter the working directory:
```sh
cd /lustre/hdd/LAS/yangli1-lab/haifengh/VidCom2/lmms-eval
```


Run the six scripts in separate tmux sessions. I didn’t use sbatch, so they’ll run interactively.
```sh
bash srun_llava_ov.sh 0.01 72b mlvu_dev,longvideobench_val_v,mvbench

bash srun_llava_ov_original.sh 72b mlvu_dev,longvideobench_val_v,mvbench

bash srun_others_llava_ov.sh vidcom2 0.01 72b mlvu_dev,longvideobench_val_v,mvbench

bash srun_others_llava_ov.sh fastvid 0.01 72b longvideobench_val_v,mvbench

bash srun_others_llava_ov.sh visionzip 0.01 72b longvideobench_val_v,mvbench

bash srun_others_llava_ov.sh prunevid 0.01 72b longvideobench_val_v,mvbench
```

Logs are saved in: `/lustre/hdd/LAS/yangli1-lab/haifengh/VidCom2/lmms-eval/logs`