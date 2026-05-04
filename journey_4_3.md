第 1 步：保留 GSM8K Dataset，确认 128 样本 SFT 能稳定跑完

第 2 步：把 max_examples 改成参数，跑 128/256/512/1024/full

第 3 步：写 eval 脚本，对每个 checkpoint 算 validation accuracy

第 4 步：画 validation accuracy curve

第 5 步：最后再加 wandb

第 6 步：如果必须严格贴合作业要求，再把 eval 从普通 generate 换成 vLLM